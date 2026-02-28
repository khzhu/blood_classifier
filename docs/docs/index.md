# blood_classifier documentation!

## Description

A non-invasive classifier predicts Isocitrate DeHydrogenase mutation status in ctDNA methylation profiles analyzed with Illumina Infinium Epic arrays.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `az storage blob upload-batch -d` to recursively sync files in `data/` up to `blood_classifier/data/`.
* `make sync_data_down` will use `az storage blob upload-batch -d` to recursively sync files from `blood_classifier/data/` to `data/`.


