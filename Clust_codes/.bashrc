# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

alias cdw='cd ~/git/SlaveBoson/clust3AB/'
alias op='source ~/basEnv/bin/activate'
alias ml-base='ml load GCCcore/11.2.0 Python/3.9.6 git/2.33.1-nodocs'
alias myqu='squeue -u rossid'
