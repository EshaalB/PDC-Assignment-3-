DIRS = cs1_heat_mpi cs2_kmer_openmp cs3_montecarlo cs4_pagerank_openmp cs5_samplesort_mpi
GROUP ?= GroupXX

.PHONY: all clean zip $(DIRS)

all: $(DIRS)

$(DIRS):
	$(MAKE) -C $@

clean:
	for d in $(DIRS); do $(MAKE) -C $$d clean; done

zip: clean
	rm -f $(GROUP)_Assignment3.zip
	mkdir -p /tmp/$(GROUP)_Assignment3
	cp -r $(DIRS) report slides CONTRIBUTIONS.md README.md /tmp/$(GROUP)_Assignment3/
	cd /tmp && zip -r $(CURDIR)/$(GROUP)_Assignment3.zip $(GROUP)_Assignment3
	rm -rf /tmp/$(GROUP)_Assignment3
