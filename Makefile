# Make changes to the variables below
pubtabnet_dir := "<your path to pubtabnet>/pubtabnet"
######################
SHELL := /bin/bash
VENV_NAME := adp
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(VENV_NAME)
PYTHON := $(CONDA_ACTIVATE) && python
PIP := $(CONDA_ACTIVATE) && pip3
# Stacked single-node multi-worker: https://pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker 
TORCHRUN = $(CONDA_ACTIVATE) && torchrun --rdzv-backend=c10d --rdzv_endpoint localhost:0 --nnodes=1 --nproc_per_node=$(NGPU)

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

#
# Virtual Environment Targets
#
clean:
> rm -f .venv_done

.venv_done: clean
> conda create -n $(VENV_NAME) python=3.9
> $(PIP) install -r requirements.txt
> $(PIP) install -e .
> touch $@

#
# Python Targets
#
include MODEL.mk
SRC := src
BEST_MODEL = "../$*/model/best.pt"
RESULT_JSON := html_table_result.json
TEDS_STRUCTURE = -f "../experiments/$*/$(RESULT_JSON)" -s

# Experiment Configurations
EXP_r18_e2_d4_adamw := $(PUBTABNET) $(MODEL_r18_e2_d4) $(OPT_adamw) ++trainer.train.batch_size=64 ++trainer.valid.batch_size=64
EXP_r34_e2_d4_adamw := $(PUBTABNET) $(MODEL_r34_e2_d4) $(OPT_adamw) ++trainer.train.batch_size=64 ++trainer.valid.batch_size=32
EXP_r50_e2_d4_adamw := $(PUBTABNET) $(MODEL_r50_e2_d4) $(OPT_adamw) ++trainer.train.batch_size=32 ++trainer.valid.batch_size=16

EXP_p14_e4_d4_nhead8_adamw := $(PUBTABNET) $(MODEL_p14_e4_d4_nhead8) $(OPT_adamw)
EXP_p16_e4_d4_nhead8_adamw := $(PUBTABNET) $(MODEL_p16_e4_d4_nhead8) $(OPT_adamw)
EXP_p28_e4_d4_nhead8_adamw := $(PUBTABNET) $(MODEL_p28_e4_d4_nhead8) $(OPT_adamw)
EXP_p56_e4_d4_nhead8_adamw := $(PUBTABNET) $(MODEL_p56_e4_d4_nhead8) $(OPT_adamw)
EXP_p112_e4_d4_nhead8_adamw := $(PUBTABNET) $(MODEL_p112_e4_d4_nhead8) $(OPT_adamw)

EXP_cs_c384_e4_d4_nhead8_adamw := $(PUBTABNET) $(MODEL_cs_c384_e4_d4_nhead8) $(OPT_adamw) ++trainer.train.batch_size=64
EXP_cs_c384_k5_e4_d4_nhead8_i476_adamw := $(PUBTABNET) $(MODEL_cs_c384_k5_e4_d4_nhead8_i476) $(OPT_adamw) ++trainer.train.batch_size=64
EXP_cs_c192_p8_k5_e4_d4_nhead8_i224_adamw := $(PUBTABNET) $(MODEL_cs_c192_p8_k5_e4_d4_nhead8_i224) $(OPT_adamw) ++trainer.train.batch_size=64
EXP_cs_c384_e4_d4_nhead8_i252_adamw := $(PUBTABNET) $(MODEL_cs_c384_e4_d4_nhead8_i252) $(OPT_adamw)
EXP_cs_c384_k5_e4_d4_nhead8_i392_adamw := $(PUBTABNET) $(MODEL_cs_c384_k5_e4_d4_nhead8_i392) $(OPT_adamw)
EXP_cs_c384_k5_e4_d4_nhead8_i504_adamw := $(PUBTABNET) $(MODEL_cs_c384_k5_e4_d4_nhead8_i504) $(OPT_adamw) ++trainer.train.batch_size=64
EXP_cs_c384_k5_e4_d4_nhead8_adamw := $(PUBTABNET) $(MODEL_cs_c384_k5_e4_d4_nhead8) $(OPT_adamw) ++trainer.train.batch_size=64

######################
NGPU := 1

.SECONDARY:

experiments/%/.done_train_structure:
> @echo "Using experiment configurations from variable EXP_$*"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="train"
> touch $@

experiments/%/.done_test_structure: experiments/%/.done_train_structure
> @echo "Using experiment configurations from variable EXP_$*"
> cd $(SRC) && $(TORCHRUN) -m main ++name=$* $(EXP_$*) ++trainer.mode="test" ++trainer.test.model=$(BEST_MODEL)
> touch $@

experiments/%/.done_teds_structure: experiments/%/.done_test_structure
> cd $(SRC) && $(PYTHON) -m utils.teds $(TEDS_STRUCTURE)
> touch $@
