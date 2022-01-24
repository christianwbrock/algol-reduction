
NORM=normalize_spectrum @norm_betCep.args --dont-plot
VOIGT=fit_three_voigts @voigt_betCep.args --dont-plot
PHASE=display_voigt_fit @phase_betCep.args
# PHASE=display_voigt_fit @phase_betCep.args --dont-plot

INPUT_FILES = $(wildcard *-noh2o.fit)
NORMALIZED_FILES = $(patsubst %-noh2o.fit,%-noh2o-norm.fit,$(INPUT_FILES))
RESULT_FILES = $(patsubst %-noh2o.fit,%.res,$(INPUT_FILES))

RESULT_TIME = absorption.csv
RESULT_PHASE = by_phase.csv

all:: $(RESULT_PHASE)

clean::
	$(RM) $(RESULT_FILES) $(NORMALIZED_FILES)

display_norm: $(NORMALIZED_FILES)
	fits_display1d --merge $(NORMALIZED_FILES)

$(RESULT_TIME): $(RESULT_FILES)
	$(RM) $@
	cat $^ | sort > $@

$(RESULT_PHASE): $(RESULT_TIME) phase_betCep.args
	$(RM) $@
	$(PHASE) $< --store-csv $@

%.res: %-noh2o-norm.fit voigt_betCep.args
	$(RM) $@
	$(VOIGT) $< --store-csv $@

%-noh2o-norm.fit: %-noh2o.fit norm_betCep.args
	$(RM) $@
	$(NORM) $< --store-fits $@

