MODULE_TOPDIR = ../../..

PGM = m.neural_network

include $(MODULE_TOPDIR)/include/Make/Other.make
include $(MODULE_TOPDIR)/include/Make/Python.make

MODULES = smp_train smp_test smp_inference __init__

ETCDIR = $(ETC)/$(PGM)/smp_lib

PYFILES := $(patsubst %,$(ETCDIR)/%.py,$(MODULES))

default: $(PYFILES)

$(ETCDIR):
	$(MKDIR) $@

$(ETCDIR)/%: % | $(ETCDIR)
	$(INSTALL_DATA) $< $@

install:
	$(MKDIR) $(INST_DIR)/etc/$(PGM)/smp_lib
	cp -r $(ETCDIR) $(INST_DIR)/etc/$(PGM)
