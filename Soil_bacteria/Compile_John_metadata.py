#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:16:07 2022

Compile serotonin qunit mutant vs LSD metadata 

@author: tobrien
"""
# %%
import pandas as pd
from pathlib import Path


from tierpsytools.hydra.match_wells_annotations import (
    update_metadata_with_wells_annotations)

from tierpsytools.hydra.compile_metadata import (
    populate_96WPs, get_source_metadata, merge_basic_and_source_meta,
    get_day_metadata, concatenate_days_metadata)

#%%
if __name__ == '__main__':

    # where are things
    proj_root = Path('/Volumes/behavgenom$/John/data_exp_info/ODonnell_lib/data')
    aux_root_dir = proj_root / 'AuxiliaryFiles'
    imaging_days = [
                    '20240710', 
                    '20240720',
                    '20240816',
                    '20240830'
                    ]

    # source plates is the same every day, but not all plates were imaged every day
    sourceplate_file = aux_root_dir / f'source_plates.csv'
    source2imaging_file = aux_root_dir / f'source2imaging.csv'

    # data for checks:
    n_plates_imaged_each_day = {
        '20240710': 7,
        '20240720': 7,
        '20240816': 12,
        '20240830': 5
    }

    # since we're using a single sourceplate and source2imaging file all days,
    # I can join them once only.
    # I'll have to then deal with any plate that did not get imaged in the loop

    # join info about drugs with name of imaging plate
    source_metadata = get_source_metadata(
        sourceplate_file, source2imaging_file)
    # checks:
    # # Ideally, 5 plates imaged each day. Same source plate replicated 4 times
    # # and 5 unique values for the imaging_plate_id
    assert source_metadata['imaging_plate_id'].nunique() == 12
    # # but only 2 unique values for source_plate_id
    assert source_metadata['source_plate_id'].nunique() == 2


# %%
    for date in imaging_days:

        print(f'Processing {date}')

        # where are things
        day_dir = aux_root_dir / date
        manual_meta_file = day_dir / f'{date}_manual_metadata.csv'
        assert manual_meta_file.exists()
        wormsorter_file = day_dir / f'{date}_wormsorter.csv'
        assert wormsorter_file.exists()
        # output file
        metadata_file = day_dir / f'{date}_day_metadata.csv'

        # how many plates were imaged in this day?
        n_img_plts = n_plates_imaged_each_day[date]

        # expand the wormsorter file to have 1 row per well
        plate_metadata = populate_96WPs(
            wormsorter_file,
            saveto=None,
            del_if_exists=False)

        # checks:
        assert plate_metadata.shape[0] % 96 == 0, 'df not multiple of 96'
        # I know how many plates where imaged today
        assert plate_metadata['imaging_plate_id'].nunique() == n_img_plts
        # each plate was filled with worms so I'm expecting plate_metadata
        # to have n_img_plts plates * 96 wells rows
        assert plate_metadata.shape[0] == n_img_plts*96

        # join worms and drugs information
        complete_plate_metadata = merge_basic_and_source_meta(
            plate_metadata,
            source_metadata,
            merge_on=['imaging_plate_id', 'well_name'],
            )
        # this was an outer merge, and not all imaging_plate_id in the source
        # would have appeared in the plate_metadata.
        # Clean up by dropping the rows where worm_strain is nan
        # (worm_strain really should not be nan in the plate_metadata)
        complete_plate_metadata = complete_plate_metadata.dropna(
            subset=['worm_strain'])
        # checks:
        assert complete_plate_metadata.shape[0] == n_img_plts*96
        assert complete_plate_metadata[
            'imaging_plate_id'].nunique() == n_img_plts
        assert complete_plate_metadata.shape[0] == plate_metadata.shape[0]
        # can't easily check how many imagin_plates map to each source bc some
        # were not imaged

        # now we join the worm/drug info of the physical plates
        # with imaging information
        day_metadata = get_day_metadata(
            complete_plate_metadata,
            manual_meta_file,
            saveto=metadata_file,
            del_if_exists=True,
            include_imgstore_name=True,
            run_number_regex=r'run\d+_')
        # checks:
        # we expect the output to be 3x as long as the input because of
        # prestim/bluelight/poststim videos
        # assert day_metadata.shape[0] == 3 * complete_plate_metadata.shape[0]

    # now we concatenate all the days metadata
    all_meta = concatenate_days_metadata(aux_root_dir, list_days=imaging_days)
    # checks:
    # assert all_meta.shape[0] % 3 == 0
    # for dd, mm in all_meta.groupby('date_yyyymmdd'):
    #     sd = str(dd)
    #     assert mm['imaging_plate_id'].nunique() == n_plates_imaged_each_day[sd]
 #print(len(all_meta))



#%%
    # # and add the information from the wells_annotations.hdf5 files
    wells_annotated_metadata = update_metadata_with_wells_annotations(
        aux_root_dir, saveto=None, del_if_exists=True)

    wells_annotated_metadata.to_csv(
        aux_root_dir / 'wells_updated_metadata.csv', index=False)


# %%
wells_annotated_metadata = update_metadata_with_wells_annotations(aux_root_dir, saveto=None, del_if_exists=True)
if wells_annotated_metadata is not None:
    wells_annotated_metadata.to_csv(aux_root_dir / 'wells_updated_metadata.csv', index=False)
else:
    print("No data to save.")
# %%
