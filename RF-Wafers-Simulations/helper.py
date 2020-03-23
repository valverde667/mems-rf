import subprocess

def gitversion():
    """Return the current git version

    Should also check for local changes and indicate them somehow
    """
    version = subprocess.check_output(['git', 'describe', '--always', '--dirty'])
    version = version.strip().decode('ascii')
    return "Version: git-" + version

