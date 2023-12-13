#!/usr/bin/env python3
import argparse
import subprocess
import os
import time
import re
import logging
import yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def configLogger(name, loglevel=logging.INFO):
    # define a Handler which writes INFO messages or higher to the sys.stderr
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    console.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(console)


logger = logging.getLogger('TransferData')
configLogger('TransferData')


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def check_grid_proxy(verbose=False, retry=3):
    import subprocess
    retry_count = 0
    while True:
        retry_count += 1
        if retry_count > retry:
            raise RuntimeError('Failed to set up valid grid proxy')
        p = subprocess.Popen('voms-proxy-info -exists', shell=True)
        p.communicate()
        if p.returncode == 0:
            if verbose:
                logging.info('Grid proxy is valid:')
                p = subprocess.Popen('voms-proxy-info', shell=True)
                p.communicate()
            break
        else:
            if verbose:
                logging.info('No valid grid proxy, will run `voms-proxy-init -rfc -voms cms -valid 192:00`.')
            p = subprocess.Popen('voms-proxy-init -rfc -voms cms -valid 192:00', shell=True)
            p.communicate()


def make_filelist(inputdir, output=None):
    filelist = []
    for dp, _, filenames in os.walk(inputdir):
        if 'failed' in dp:
            continue
        for f in filenames:
            if not f.endswith('.root'):
                continue
            if os.path.getsize(os.path.join(dp, f)) < 1000:
                logger.warning('Ignoring empty file %s' % os.path.join(dp, f))
                continue
            filelist.append(os.path.join(dp, f))
    filelist = natural_sort(filelist)
    if output is not None:
        with open(output, 'w') as f:
            yaml.safe_dump({'files': filelist}, f, sort_keys=True)
    return filelist


def load_filelist(filepath):
    filelist = []
    with open(filepath) as f:
        filelist = yaml.safe_load(f)['files']
    return filelist


def copy(src, dst):
    cmd = f'xrdcp --silent -f -P {src} {dst}'
    # logger.info(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.communicate()
    if p.returncode != 0:
        logger.error(f'Failed: {src}')
        return src


def main():

    parser = argparse.ArgumentParser('Transfer datasets.')
    parser.add_argument('-i', '--inputdir',
                        help='Input dir.'
                        )
    parser.add_argument('-o', '--outputdir',
                        help='Destination directory.'
                        )
    parser.add_argument('--src-server', default='root://cmseos.fnal.gov/',
                        help='Source connect string.'
                        )
    parser.add_argument('--dst-server', default='root://eoscms.cern.ch/',
                        help='Destination connect string.'
                        )
    parser.add_argument('--exclude',
                        help='Exclude files in this list.'
                        )
    parser.add_argument('--make-filelist', default=None,
                        help='Make file list.'
                        )
    parser.add_argument('-n', '--num-cores', default=8, type=int,
                        help='Number of cores to use.'
                        )
    parser.add_argument('--dryrun', action='store_true',
                        help='Only list the files to be transfer.'
                        )
    args = parser.parse_args()

    if args.make_filelist:
        make_filelist(args.inputdir, args.make_filelist)
        return

    check_grid_proxy()

    filelist = make_filelist(args.inputdir)
    logger.info(f'Found {len(filelist)} files in the inputdir')
    exclude = set()
    if args.exclude:
        exclude = set(f.replace(args.outputdir, '') for f in load_filelist(args.exclude))

    params = []
    transfer_info = {}
    for f in filelist:
        base_path = f.replace(args.inputdir, '')
        if base_path in exclude:
            continue
        src = args.src_server + f
        dst = args.dst_server + args.outputdir + base_path
        params.append((src, dst))

        # for information
        dataset = ''
        dirnames = f.split('/')
        for i, n in enumerate(dirnames):
            if 'TeV' in n or 'Tune' in n:
                dataset = n
                break
            r = re.search(r'(Run20[0-9]+[A-Z])', n)
            if r:
                if i > 0:
                    dataset = dirnames[i - 1] + '/' + r.groups()[0]
                else:
                    dataset = n
                break

        try:
            transfer_info[dataset] += 1
        except KeyError:
            transfer_info[dataset] = 1

    logger.info(
        f'Will transfer {len(params)} files in total:\n - ' + '\n - '.join([str(it) for it in transfer_info.items()]))

    if args.dryrun:
        return

    results = []
    with ProcessPoolExecutor(max_workers=args.num_cores) as pool:
        with tqdm(total=len(params)) as progress:
            futures = []
            for src, dst in params:
                future = pool.submit(copy, src, dst)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)

    logger.info(f'{len(results)} failed: \n' + '\n-'.join(results))


if __name__ == '__main__':
    main()
