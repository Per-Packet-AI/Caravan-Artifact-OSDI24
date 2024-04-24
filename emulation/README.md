# Caravan Software Emulation

Here we provide the code and instructions to run the emulation-based results for Caravan. Below, we walk through how to set up the environment, how to automatically run the experiments, and how to plot the key figures in the paper for artifact evaluation purposes.

## Getting Started 

To evaluate our artifact, we will give the artifact reviewers access to one of our GCP instances. **Please find the account info that can be used to access the GCP instance on HotCRP.** We appreciate it if the reviewers could coordinate amongst themselves to share the instance.

### Accessing the machine:
The recommended option is to use the [gcloud CLI](https://cloud.google.com/sdk/docs/install).
Once installed, the user must first log in:
```
gcloud auth login
```
After logging in, make sure that the SSH connection is made using the `caravan` user. If needed, the instance should be started first.
```
gcloud compute instances start "caravan-artifact" --zone "us-central1-a" --project "soe-kunle-research-gcp"
```

```
gcloud compute ssh --zone "us-central1-a" "caravan@caravan-artifact" --project "soe-kunle-research-gcp"
```

<!-- **Logging into the machine:** Please create a file called `osdi24_ae.pem` and copy the SSH private key (from HotCRP) to the file. After that, you can access the GCP instance through
```
chmod 400 osdi24_ae.pem
ssh -i "osdi24_ae.pem" ubuntu@ec2-18-222-217-245.us-east-2.compute.amazonaws.com
``` -->

Please set the environment variables and install the dependencies (you can skip this step if you are running the GCP instance we provide)
```
export CARAVAN_HOME=$HOME/Caravan;
cd $CARAVAN_HOME/;
git pull;
cd $CARAVAN_HOME/emulation/;
pip install -e .
```

We also provide the artifact reviewers with an OpenAI API key for running experiments that involve GPT3.5/GPT4. **Please find this OpenAI API key on HotCRP.** We appreciate if you keep this key private/confidential. After you have the key, set it as an environment variable (you can skip this step if you are running the GCP instance we provide)
```
export OPENAI_API_KEY=sk-******
```

## Directory Structure

The emulation folder is mainly composed of three sub-folders:
* `benchmarks/`: contains the benchmarks written for experiment purposes, including the continuous training pipeline, the labeling rule cache logic, the retraining trigger logic, etc.
* `datasets/`: contains the datasets we use for evaluation purposes.
* `models/`: contains the DNN models (in-network ML models and labeler DNN models) we use for evaluation purposes.
* `scripts/`: contains the scripts to automatically run the experiments and plot the figures.
* `src/`: contains the implementation of Caravan and its different components, including the architecture of in-network ML models, the labeling agent, the streaming dataset constructor, etc.

## Kick-the-Tire Experiment

As a warm-up (during kick-the-tire period), you will run the experiment associated with `Figure 4a` and generate a plot. Please follow the commands below:
```
python3 scripts/experiments/figure-4a.py;
python3 scripts/plots/figure-4a.py
```
This should automatically run the experiment and plot out `Figure 4a` (stored as `scripts/figures/figure-4a.png`) in about two minutes.

> **FAQs:**
> * **Why is my figure slightly different from the one in the paper?** In the paper, the results are the average of 5--10 different trials, each one using a different random seed. As a result, it is likely that you will get a similar (but not exactly the same) plot for most experiments. However, patterns in the plots should follow the discussed trends, regardless. Another source of randomness is the use of GPT4 APIs. We find that even when using the same model snapshot and the same prompts, the response from the GPT4 API and the latency of a single GPT4 request can vary significantly across time. 
> * **What if something does not work?** If you are unable to SSH into the GCP instance we provide, or if you run into any trouble during the kick-the-tire experiment or with the OpenAI API key, please feel free to contact us on HotCRP and we are happy to help.

## Running Experiments

To run all the experiments at once, please run `./scripts/experiments.sh`. The intermediate results will be stored under `scripts/results/`. Please don't delete the results folder as its needed for plotting in the next step. We recommend using `tmux` since the GPT4 experiments can take a while.

To run individual experiments, e.g. when the `experiment.sh` script breaks in the middle, please refer to the table below (with ETAs):

| Experiment  | Script        | ETA |
| ----------- | ----------- | -----------  |
| Figure 4a   | scripts/experiments/figure-4a.py    |  2 minutes |
| Figure 4b   | scripts/experiments/figure-4b.py    |  121 minutes | 
| Figure 5   | scripts/experiments/figure-5.py    |  2 minutes | 
| Figure 6   | scripts/experiments/figure-6.py    |  87 minutes | 
| Figure 7   | scripts/experiments/figure-7.py    |  2 minutes | 
| Figure 8   | scripts/experiments/figure-8.py    |  2 minutes | 
| Figure 9   | scripts/experiments/figure-9.py    |  5 minutes | 
| Figure 10a   | scripts/experiments/figure-10a.py    | 6 minutes | 
| Figure 10c   | scripts/experiments/figure-10c.py    | 20 minutes | 

## Plotting Figures

To automatically plot all figures, please run `./scripts/plots.sh`. The figures will be stored under `scripts/figures/`. Similar to the experiments, you can also plot individual figures using the scripts in `scripts/plots`.
