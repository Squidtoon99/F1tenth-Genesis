module load apptainer/1.3.4

if [ ! -d ~/scratch/redis ]; then
    mkdir ~/scratch/redis
fi
cd ~/scratch/redis
apptainer exec docker://redis redis-server