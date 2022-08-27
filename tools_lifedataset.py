"""
Functions specific for LIFE data set.
"""
import os
import numpy as np
import mne
import pandas as pd
from tools_general import load_json


def read_erp(subj, dir_data, decim=1, notch=False, h_freq=45):
    """
    Read the data from stimulus based recordings

    Adapted from Denis Engemann <denis.engemann@gmail.com>

    :param str subj: id of a participant
    :param str dir_data: directory with data files
    :param int decim: decimation factor, default decim = 1 (no decimation)
    :param bool notch: application of the notch filter around 50 Hz, default = False
    :param int h_freq: frequency of cut-off, default = 45 Hz
    :returns: erp_s, erp_t, erp_n (instance of mne.Epochs) - Epochs of three types of stimuli
    """

    from tools_external import _get_global_reject_epochs
    from tools_signal import project_eog

    files_erp = load_json('settings/files_erp', os.getcwd())
    data_erp_file = dir_data + files_erp[subj]
    erp = mne.io.read_raw_brainvision(data_erp_file, eog=['VEOG', 'HEOG'], misc=['EKG'])

    # add montage
    montage = mne.channels.make_standard_montage('easycap-M1')
    erp.load_data()
    erp.set_montage(montage)
    # average reference
    erp.set_eeg_reference()
    # broadband filter
    erp.filter(0.1, h_freq)
    if notch:  # default is False
        erp.notch_filter(50, picks='eeg')

    # ignore all non-stimuli annotations
    erp.annotations.description[:] = [
        'BAD' if not dd[-1].isdigit() else dd for dd in erp.annotations.description]

    # apply eye movements correction and rejection of noisy epochs
    project_eog(erp, reject_with_threshold=False, decim=5)
    reject = _get_global_reject_epochs(erp, decim=5)

    # create epochs based on annotations
    events = mne.events_from_annotations(erp)
    epochs = mne.Epochs(
        raw=erp, events=events[0], event_id=events[1],
        baseline=(-.2, -.05), reject=reject,
        reject_by_annotation=True,
        proj=True, event_repeated='drop',
        tmin=-.4, tmax=1.3, preload=True, decim=decim)

    # extract stimulus related epochs
    evokeds = [epochs[cc] for cc in epochs.event_id
               if cc in ['Stimulus/S 10', 'Stimulus/S 20', 'Stimulus/S 30']]

    erp_s = evokeds[0]  # after standard stimulus
    erp_t = evokeds[1]  # after target stimulus
    erp_n = evokeds[2]  # after novelty stimulus

    return erp_s, erp_t, erp_n


def create_raw_for_source_reconstruction(raw, decim=5):
    """
    Prepare the data for source reconstruction

    :param mne.Raw raw: data to prepare
    :param int decim: decimation factor, default decim = 5
    :returns: evoked_raw (mne.EvokedArray) - data that can be passed to source reconstruction
    """

    # create new info without eog and acg channels
    raw_info = raw.copy().info
    raw_info['ch_names'] = np.asarray(raw.info['ch_names'][:-3]).tolist()
    raw_info['chs'] = np.asarray(raw.info['chs'][:-3]).tolist()
    raw_info['nchan'] = len(raw_info.ch_names)

    # create new raw
    raw_data = raw.get_data(picks='eeg')
    evoked_raw = mne.EvokedArray(raw_data, raw_info, tmin=0, nave=1)
    evoked_raw.decimate(decim)
    evoked_raw.set_eeg_reference(projection=True)

    return evoked_raw


def create_erp_for_source_reconstruction(erp, decim=10):
    """
    Prepare the data for source reconstruction

    :param mne.Epochs erp: data to prepare
    :param int decim: decimation factor, default decim = 10
    :returns: evoked_erp (mne.EvokedArray) - data that can be passed to source reconstruction
    """

    from tools_signal import from_epoch_to_cont

    # extract the data
    erp.decimate(decim)
    erp_data = erp.get_data()

    # if data array is not 3D - return an error
    if len(erp_data.shape) != 3:
        raise ValueError('Input data should be epoched.')

    epochs, channels, times = erp_data.shape
    # check if first dimension is epochs
    if epochs != len(erp.events):
        epoch_dim = np.where(np.array(erp_data.shape) == len(erp.events))[0][0]
        erp_data = np.swapaxes(erp_data, epoch_dim, 0)
        epochs = len(erp.events)
    # check if second dimension is channels
    if channels > times:
        erp_data = np.swapaxes(erp_data, 2, 1)
        channels, times = times, channels

    # flatten the data
    erp_full = from_epoch_to_cont(erp_data, channels, epochs)

    # create evoked array for further reconstruction
    evoked_erp = mne.EvokedArray(erp_full, erp.info, tmin=0, nave=erp_full.shape[0])
    evoked_erp.set_eeg_reference(projection=True)

    return evoked_erp


def read_rest(subj, dir_data, marker):
    """
    Read the data from resting-state recordings

    :param str subj: id of a subject
    :param str dir_data: directory with data files
    :param float marker: value of starting point
    :returns: raw (mne.Raw) - preprocessed resting-state data
    """

    # read file with raw rest data
    raw = read_raw_rest(subj, dir_data)
    # read file with cleaned rest data (Cesnaite et al., 2021)
    cleaned = read_cleaned_rest(subj, dir_data)

    # extract bad segments and bad channels for cleaned and apply it to raw
    if type(cleaned) == list:  # this subject doesn't have markers from cleaned data
        new_description = raw.annotations.description
        new_onset = raw.annotations.onset + marker
        new_duration = raw.annotations.duration
        sorted_idx = np.argsort(new_onset)
    else:
        new_description = np.hstack((raw.annotations.description, cleaned.annotations.description))
        new_onset = np.hstack((raw.annotations.onset, cleaned.annotations.onset + marker))
        new_duration = np.hstack((raw.annotations.duration, cleaned.annotations.duration))
        sorted_idx = np.argsort(new_onset)
    # new annotations
    new_annot = mne.Annotations(onset=new_onset[sorted_idx],
                                duration=new_duration[sorted_idx],
                                description=new_description[sorted_idx],
                                orig_time=raw.annotations.orig_time)
    raw.set_annotations(new_annot)

    # filter in a broadband
    raw.filter(0.1, 45)
    raw.notch_filter(50, picks='eeg')
    # treat all annotation except start, break, and end as artifactual
    raw.annotations.description[:] = ['BAD' if not dd[-1].isdigit() else dd
                                      for dd in raw.annotations.description]
    # cut out bad segments
    raw_cut = raw.copy()
    raw_data = raw_cut.get_data()
    fs = raw.info['sfreq']
    for i_ann in range(len(raw.annotations) - 1, -1, -1):
        if raw.annotations.description[i_ann] == 'BAD':
            start = int(raw.annotations.onset[i_ann] * fs)
            end = int((raw.annotations.onset[i_ann] + raw.annotations.duration[i_ann]) * fs)
            raw_data = np.delete(raw_data, np.s_[start:end], axis=1)
    raw._data = raw_data

    # take first 10 min of the recording
    try:
        raw.crop(tmin=marker, tmax=marker + 600)
    except:
        print('The file is too short. Taking all of it.')

    return raw


def read_raw_rest(subj, dir_data):
    """
    Read the data from rest recordings

    :param str subj: id of a participant
    :param str dir_data: directory with data files
    :returns: raw (mne.Raw) - Raw instance with rest signals
    """

    files_rest = load_json('settings/files_rest', os.getcwd())

    data_raw_file = dir_data + files_rest[subj]
    raw = mne.io.read_raw_brainvision(data_raw_file, eog=['VEOG', 'HEOG'], misc=['EKG'])

    # read montage and apply
    montage = mne.channels.make_standard_montage('easycap-M1')
    raw.set_montage(montage)
    raw.load_data()
    # set average reference
    raw.set_eeg_reference()

    return raw


def read_cleaned_rest(subj, dir_data):
    """
    Read the data from rest recordings

    :param str subj: id of a participant
    :param str dir_data: directory with data files
    :returns: cleaned (mne.Raw) - Raw instance with rest signals
    """

    files_cleaned_rest = load_json('settings/files_clean_rest', os.getcwd())
    try:
        data_cleaned_file = dir_data + files_cleaned_rest[subj]
        cleaned = mne.io.read_raw_eeglab(data_cleaned_file, preload=True)
        # set average reference
        cleaned.set_eeg_reference()
        return cleaned
    except ValueError:  # only one subject
        print(subj + ' does not have clean data set')
        return []


def read_age(ids):
    """
    Collects age of each subject

    :param list ids: ids of subjects
    :returns:
        age (numpy.ndarray, 1D) - age of each subject;
        ids_age (numpy.ndarray, 1D) - ids of those subjects for whom age is available
    """

    age_data = pd.read_excel(load_json('settings_real/dirs_files', os.getcwd())['file_age'],
                             engine='openpyxl')
    ids_age_all = age_data['TEILNEHMER_SIC'].to_numpy()
    age_all = age_data['ageEEG'].to_numpy()
    _, idx1, idx2 = np.intersect1d(ids, ids_age_all, return_indices=True)
    ids_age = np.array(ids)[idx1]
    age = age_all[idx2]

    return age, ids_age


def read_gender(ids):
    """
    Collects gender of each subject

    :param list ids: ids of subjects
    :returns:
        gender (numpy.ndarray, 1D) - gender of each subject;
        ids_gender (numpy.ndarray, 1D) - ids of those subjects for whom gender is available
    """

    gender_data = pd.read_excel(load_json('settings_real/dirs_files', os.getcwd())['file_gender'],
                                engine='openpyxl')
    ids_gender_all = gender_data['TEILNEHMER_SIC'].to_numpy()
    gender_all = gender_data['TEILNEHMER_GESCHLECHTEEG'].to_numpy()
    _, idx1, idx2 = np.intersect1d(ids, ids_gender_all, return_indices=True)
    ids_gender = np.array(ids)[idx1]
    gender = gender_all[idx2]

    return gender, ids_gender


def read_cerad(ids):
    """
    Computes scores from CERADplus: Retrieve word list, Word list recognition, Retrieve figures

    :param list ids: ids of subjects
    :returns:
        WLDR (numpy.ndarray, 1D) - scores for Retrieve word list task;
        WLR (numpy.ndarray, 1D) - scores for Word list recognition;
        FDR (numpy.ndarray, 1D) - scores for Retrieve figures task;
        cerad_ids (numpy.ndarray, 1D) - ids of those subjects for whom scores were available
    """

    file = pd.read_excel(load_json('settings_real/dirs_files', os.getcwd())['file_cerad'], engine='openpyxl')
    file = file.fillna(0)  # fills nan values with zeros

    # selects columns for Retrieve word list task
    cerad_WLDR_all = file[
        ['CERAD_WL4_BUTTER', 'CERAD_WL4_ARM', 'CERAD_WL4_STRAND', 'CERAD_WL4_BRIEF',
         'CERAD_WL4_KONIGI', 'CERAD_WL4_HUTTE', 'CERAD_WL4_STANGE', 'CERAD_WL4_KARTE',
         'CERAD_WL4_GRAS', 'CERAD_WL4_MOTOR']].to_numpy()
    # selects columns for Word list recognition
    cerad_WLR_all = file[
        ['CERAD_WLW_KIRCHE', 'CERAD_WLW_KAFFEE', 'CERAD_WLW_BUTTER', 'CERAD_WLW_DOLLAR',
         'CERAD_WLW_ARM', 'CERAD_WLW_STRAND', 'CERAD_WLW_FUNF', 'CERAD_WLW_BRIEF',
         'CERAD_WLW_HOTEL', 'CERAD_WLW_BERG', 'CERAD_WLW_KONIGI', 'CERAD_WLW_HUTTE',
         'CERAD_WLW_PANTOF', 'CERAD_WLW_STANGE', 'CERAD_WLW_DORF', 'CERAD_WLW_BAND',
         'CERAD_WLW_KARTE', 'CERAD_WLW_HEER', 'CERAD_WLW_GRAS',
         'CERAD_WLW_MOTOR']].to_numpy(dtype='float')
    # selects columns for Retrieve figures task
    cerad_FDR_all = file[
        ['CERAD_FIGW_1A', 'CERAD_FIGW_1B', 'CERAD_FIGW_2A', 'CERAD_FIGW_2B',
         'CERAD_FIGW_2C', 'CERAD_FIGW_3A', 'CERAD_FIGW_3B', 'CERAD_FIGW_4A',
         'CERAD_FIGW_4B', 'CERAD_FIGW_4C', 'CERAD_FIGW_4D', 'CERAD_FIGW_5A',
         'CERAD_FIGW_5B', 'CERAD_FIGW_5C']].to_numpy(dtype='float')

    cerad_all_ids = file['CERAD_SIC'].to_numpy()
    cerad_ids, _, idx2 = np.intersect1d(ids, cerad_all_ids, return_indices=True)

    WLDR = np.mean(cerad_WLDR_all, axis=1)[idx2]  # resulting score is average over all answers

    # implausible answers set to 0
    for i in range(len(cerad_all_ids)):
        if np.mean(cerad_WLR_all[i]) > 1:
            for j in range(len(cerad_WLR_all[i])):
                if cerad_WLR_all[i, j] > 1:
                    cerad_WLR_all[i, j] = 0 * cerad_WLR_all[i, j]
    WLR = np.mean(cerad_WLR_all, axis=1)[idx2]  # resulting score is average over all answers

    # implausible answers set to 0
    for i in range(len(cerad_all_ids)):
        if np.mean(cerad_FDR_all[i]) > 1:
            for j in range(len(cerad_FDR_all[i])):
                if cerad_FDR_all[i, j] > 1:
                    cerad_FDR_all[i, j] = 0 * cerad_FDR_all[i, j]
    FDR = np.mean(cerad_FDR_all, axis=1)[idx2]  # resulting score is average over all answers

    return WLDR, WLR, FDR, cerad_ids


def composite_attention(ids):
    """
    Computes scores for composite attention from TMT-A time-to-complete and
    Stroop neutral time-to-complete

    :param list ids: ids of subjects
    :returns:
        attention_composite (numpy.ndarray, 1D) - composite scores for memory;
        attention_ids (numpy.ndarray, 1D) - ids of those subjects for whom scores have been computed
    """
    from tools_general import scaler_transform

    # TMT-A
    file = pd.read_excel(load_json('settings_real/dirs_files', os.getcwd())['file_tmt'], engine='openpyxl',
                         na_values=np.nan)

    # extract values and ids from the file
    tmta_time_all = file[['TMT_TIMEA']].to_numpy()
    tmt_ids_all = file['TMT_SIC'].to_numpy()

    # collect implausible scores' indices
    impl_idx = []
    impl_idx.append(list(np.where(tmta_time_all.reshape((-1)) == 997.)[0]))  # not applicable
    impl_idx.append(list(np.where(tmta_time_all.reshape((-1)) == 998.)[0]))  # do not know
    impl_idx.append(list(np.where(tmta_time_all.reshape((-1)) == 999.)[0]))  # refusal to answer
    impl_idx = np.array([j for idx in impl_idx for j in idx])

    # substitute implausible scores and nan values with averages
    if len(impl_idx) > 0:
        tmta_time_all[impl_idx] = np.nanmean(np.delete(tmta_time_all, impl_idx, axis=0))
        tmta_time_all[np.isnan(tmta_time_all)] = np.nanmean(np.delete(tmta_time_all,
                                                                      impl_idx, axis=0))
    else:
        tmta_time_all[np.isnan(tmta_time_all)] = np.nanmean(tmta_time_all)

    # save values only for subjects in 'ids'
    tmt_ids, _, idx2 = np.intersect1d(ids, tmt_ids_all, return_indices=True)
    tmta_time = tmta_time_all[idx2].reshape((-1))  # the smaller, the better

    # Stroop neutral
    file = pd.read_excel(load_json('settings_real/dirs_files', os.getcwd())['file_stroop'], engine='openpyxl',
                         na_values=np.nan)

    # extract values and ids from the file
    stroop_n_all = file[['STROOP_RO_RT_NEUTRAL']].to_numpy()
    stroop_all_ids = file['STROOP_RO_SIC'].to_numpy()

    # collect implausible scores' indices
    impl_idx = []
    impl_idx.append(list(np.where(stroop_n_all.reshape((-1)) == 98.)[0]))  # implausible result
    impl_idx = np.array([j for i in impl_idx for j in i])
    if len(impl_idx) > 0:
        stroop_n_all[impl_idx] = np.nanmean(np.delete(stroop_n_all, impl_idx, axis=0))
        stroop_n_all[np.isnan(stroop_n_all)] = np.nanmean(np.delete(stroop_n_all, impl_idx, axis=0))
    else:
        stroop_n_all[np.isnan(stroop_n_all)] = np.nanmean(stroop_n_all)

    # save values only for subjects in 'ids'
    stroop_ids, _, idx2 = np.intersect1d(ids, stroop_all_ids, return_indices=True)
    stroop_n = stroop_n_all[idx2].reshape((-1))  # the smaller, the better

    # find common ids for TMT and Stroop
    ids_tmt_ids, _, idx2 = np.intersect1d(ids, tmt_ids, return_indices=True)
    attention_ids, idx3, idx4 = np.intersect1d(ids_tmt_ids, stroop_ids, return_indices=True)

    # scale transform and average
    attention_all = np.vstack((tmta_time[idx2][idx3], stroop_n[idx4]))
    attention_all_scaled = scaler_transform(attention_all.T)
    attention_composite = np.mean(attention_all_scaled, axis=1)

    return attention_composite.reshape((-1)), attention_ids


def composite_memory(ids):
    """
    Computes scores for composite memory from CERADplus: Retrieve word list,
    Word list recognition, Retrieve figures

    :param list ids: ids of subjects
    :returns:
        memory_composite (numpy.ndarray, 1D) - composite scores for memory;
        memory_ids (numpy.ndarray, 1D) - ids of those subjects for whom scores have been computed
    """
    from tools_general import scaler_transform

    WLDR, WLR, FDR, cerad_ids = read_cerad(ids)  # the bigger, the better
    memory_ids, _, idx2 = np.intersect1d(ids, cerad_ids, return_indices=True)

    # scale transform and average
    memory_all = np.vstack((WLDR[idx2], WLR[idx2], FDR[idx2]))
    memory_all_scaled = scaler_transform(memory_all.T)
    memory_composite = np.mean(memory_all_scaled, axis=1)

    return memory_composite.reshape((-1)), memory_ids


def composite_executive(ids):
    """
    Computes scores for composite executive function from TMT-b time to complete
    and Stroop-incongruent time-to-complete

    :param list ids: ids of subjects
    :returns:
        executive_composite (numpy.ndarray, 1D) - composite scores for executive function;
        executive_ids (numpy.ndarray, 1D) - ids of those subjects for whom scores have been computed
    """
    from tools_general import scaler_transform

    # TMT-B
    file = pd.read_excel(load_json('settings_real/dirs_files', os.getcwd())['file_tmt'], engine='openpyxl')
    # extract values and ids from the file
    tmtb_time_all = file[['TMT_TIMEB']].to_numpy()
    tmt_ids_all = file['TMT_SIC'].to_numpy()

    # collect implausible scores' indices
    impl_idx = []
    impl_idx.append(list(np.where(tmtb_time_all.reshape((-1)) == 997.)[0]))  # not applicable
    impl_idx.append(list(np.where(tmtb_time_all.reshape((-1)) == 998.)[0]))  # do not know
    impl_idx.append(list(np.where(tmtb_time_all.reshape((-1)) == 999.)[0]))  # refusal to answer
    impl_idx = np.array([j for idx in impl_idx for j in idx])

    # substitute implausible scores and nan values with averages
    if len(impl_idx) > 0:
        tmtb_time_all[impl_idx] = np.nanmean(np.delete(tmtb_time_all, impl_idx, axis=0))
        tmtb_time_all[np.isnan(tmtb_time_all)] = np.nanmean(np.delete(tmtb_time_all,
                                                                      impl_idx, axis=0))
    else:
        tmtb_time_all[np.isnan(tmtb_time_all)] = np.nanmean(tmtb_time_all)

    # save values only for subjects in 'ids'
    tmt_ids, _, idx2 = np.intersect1d(ids, tmt_ids_all, return_indices=True)
    tmtb_time = tmtb_time_all[idx2]

    # Stroop incongruent
    file = pd.read_excel(load_json('settings_real/dirs_files', os.getcwd())['file_stroop'], engine='openpyxl')
    # extract values and ids from the file
    stroop_in_all = file[['STROOP_RO_RT_INKON']].to_numpy()
    stroop_all_ids = file['STROOP_RO_SIC'].to_numpy()
    # collect implausible scores' indices
    impl_idx = []
    impl_idx.append(list(np.where(stroop_in_all.reshape((-1)) == 98.)[0]))  # implausible result
    impl_idx = np.array([j for i in impl_idx for j in i])
    if len(impl_idx) > 0:
        stroop_in_all[impl_idx] = np.nanmean(np.delete(stroop_in_all, impl_idx, axis=0))
        stroop_in_all[np.isnan(stroop_in_all)] = np.nanmean(np.delete(stroop_in_all,
                                                                      impl_idx, axis=0))
    else:
        stroop_in_all[np.isnan(stroop_in_all)] = np.nanmean(stroop_in_all)

    # save values only for subjects in 'ids'
    stroop_ids, _, idx2 = np.intersect1d(ids, stroop_all_ids, return_indices=True)
    stroop_in = stroop_in_all[idx2]

    # find common ids for TMT and Stroop
    ids_tmt_ids, _, idx2 = np.intersect1d(ids, tmt_ids, return_indices=True)
    executive_ids, idx3, idx4 = np.intersect1d(ids_tmt_ids, stroop_ids, return_indices=True)

    # scale-transform and average
    executive_all = np.vstack((tmtb_time.reshape((-1))[idx2][idx3], stroop_in.reshape((-1))[idx4]))
    executive_all_scaled = scaler_transform(executive_all.T)
    executive_composite = np.mean(executive_all_scaled, axis=1)

    return executive_composite.reshape((-1)), executive_ids


def read_medications(ids):
    """
    Finds subjects who were under medication for depression, epilepsy, Parkinson's, dementia

    :param list ids: ids of all subjects
    :returns:
        meds_ids (numpy.ndarray, 1D) - ids of subjects to exclude;
        idx (numpy.ndarray, 1D) - indices of subjects to exclude in 'ids'
    """

    # read file with medications
    file = pd.read_excel(load_json('settings_real/dirs_files', os.getcwd())['file_meds'], engine='openpyxl')
    # columns to read
    meds_clms = file[['EEG_SUBSTANZ_I06B_1A', 'EEG_SUBSTANZ_I06B_2A', 'EEG_SUBSTANZ_I06B_3A',
                      'EEG_SUBSTANZ_I06B_4A', 'EEG_SUBSTANZ_I06B_5A', 'EEG_SUBSTANZ_I06B_6A',
                      'EEG_SUBSTANZ_I06B_7A', 'EEG_SUBSTANZ_I06B_8A', 'EEG_SUBSTANZ_I06B_9A',
                      'EEG_SUBSTANZ_I06B_10A', 'EEG_SUBSTANZ_I06B_11A', 'EEG_SUBSTANZ_I06B_12A',
                      'EEG_SUBSTANZ_I06B_13A', 'EEG_SUBSTANZ_I06B_14A',
                      'EEG_SUBSTANZ_I06B_15A']].to_numpy()
    meds_all_ids = file['EEG_SUBSTANZ_SIC'].to_numpy()
    # remove empty columns
    meds_all = [meds_clms[i][~pd.isnull(meds_clms[i])] for i in range(meds_clms.shape[0])]
    # gather unique medications
    unique_meds = []
    for med in meds_all:
        unique_meds += list(med)
    unique_meds = np.unique(unique_meds)

    # list of medications to exclude
    exclude_meds = ['amantadin al 100', 'vasikur', 'cifapopram', 'cipralex',
                    'cipramil', 'citalopram', 'citalopram al', 'citalopram hexal 50',
                    'citralopram', 'elontril', 'maprotilin', 'venlafaxin',
                    'venlafaxin?', 'yentreve', 'apydan extent', 'keppra',
                    'levetirazetam', 'phenytoin', 'pipamperon 40', 'axura',
                    'memando', 'antidepressiva valproat', 'antidrpressiva', 'ariclaim',
                    'cymbalta', 'cymbalty', 'fluxet', 'sertralin',
                    'stangyl', 'trevilor', 'fluanxol', 'seroquel prolong',
                    'sulpirid- ct', 'opipramol', 'alprazolam', 'diazepam',
                    'faustan', 'flurozepam', 'tavor', 'temgesic',
                    'triasol', 'demenzmedikament', 'exelon', 'biogamma',
                    'carbamazepin', 'carbamezapin', 'ofiril', 'trileptal',
                    'vimpat', 'levocomp', 'levodop neuraxpharm',
                    'madopar', 'matopar', 'stalevo', 'morphin',
                    'sevredol', 'fentanyl', 'amixx', 'parkinsonpflaster',
                    'requip modutab', 'ropinirol', 'sifrol', 'sormodren',
                    'tabl. zur beweglichkeit bei parkinson', 'medi. gegen depression',
                    'mirtazapin', 'amitriptilin', 'amitriptylin', 'amitryptilin',
                    'paroxetin', 'saroten', 'valdoxan', 'opipram', 'sulpril']

    # match names of medications in exclude list to how it is named in the data
    exclude_meds_case = []
    for u_med in unique_meds:
        for ex_med in exclude_meds:
            if u_med.lower() == ex_med.lower():
                exclude_meds_case.append(u_med)

    # find subjects who was under the medication at the time of testing
    exclude_meds_all = [len(set(exclude_meds_case) - set(meds_all[i])) != len(exclude_meds_case)
                        for i in range(len(meds_all))]
    meds_ids, idx, _ = np.intersect1d(ids, meds_all_ids[exclude_meds_all], return_indices=True)

    return meds_ids, idx
