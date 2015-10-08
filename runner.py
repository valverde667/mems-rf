"""

Usage:
  runner <repopath> <branch> <file> [--pagenumber=<page>]

Options:
  -h --help                      this text
  -v --version                   version information
  -p <page> --pagenumber=<page>  slice for pages, e.g. 2:200:2 or 2:200 or 2, [default: last]
"""

import concurrent.futures
import subprocess
import time
import os
import glob
import shutil
import tempfile
from collections import namedtuple
import docopt

commands = docopt.docopt(__doc__, version="1.0")

# check path
path = os.path.abspath(commands['<repopath>'])
if os.path.isdir(os.path.join(path, ".git")):
    reponame = os.path.basename(path)
    print("found git repo", reponame)
else:
    print(path, "doesn't seem to be a git repo")
    raise SystemExit

# check branch
branch = commands['<branch>']
output = str(subprocess.check_output("cd {} && git branch".format(path), shell=True))
if branch in output:
    print("found branch:", branch)
else:
    print("Couldn't find branch:", branch, " in git repo:", path)
    raise SystemExit

# get commit list
output = subprocess.check_output("cd {} && git log {}...".format(path, branch), shell=True).decode('ascii')
output = output.split('\n')
commits = [x[7:] for x in output if x.startswith('commit ')]

# remaining command line arguments
myfile = commands['<file>']
pages = commands['--pagenumber']
pages = pages.split(';')

RunResult = namedtuple('RunResult', 'runtime tmpdir')
def run(commit):
    start = time.time()
    if not os.path.exists(os.path.join('/tmp','warptmp-'+commit)):
        # create temp dir and delete it later again... no contect manager
        # available for python 2.7, so we do it by hand
        d = tempfile.mkdtemp(commit, prefix='warptmp-')
        subprocess.call("cd {} && git clone --shared {} && cd {} && git checkout {}".
                        format(d, path, reponame, commit), shell=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # remove all previous results
        subprocess.call("cd {} && cd {} && rm *cgm*".
                        format(d, reponame), shell=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        d = os.path.join('/tmp','warptmp-'+commit)
    if os.path.exists(os.path.join(d, reponame, myfile)):
        cgmfile = glob.glob('{}/*.cgm'.format(os.path.join(d, reponame)))
        # run simulations only, if no results exists
        if not len(cgmfile):
            subprocess.call("cd {} && python {}".
                            format(os.path.join(d, reponame), myfile), shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # find gist file
        cgmfile = glob.glob('{}/*.cgm'.format(os.path.join(d, reponame)))[-1]
        for i, p in enumerate(pages):
            subprocess.call("cd {} && echo gist {} -b -ps tmp{:02d}.ps {}".
                            format(os.path.join(d, reponame), cgmfile, i, p), shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        print("can't find file", myfile, " in commit:", commit)
    dt = time.time()-start
    return RunResult(dt, d)

if __name__ == '__main__':
    tmpdirs = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for c, r in zip(commits, executor.map(run, commits)):
            print("finished", c,"runtime:", r.runtime)
            tmpdirs.append(r.tmpdir)

#    print("cleanup")
#    for d in tmpdirs:
#        shutil.rmtree(d)
