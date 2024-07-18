from benchopt import BaseDataset, safe_import_context

dataset_list = []
dataset_list_str = []
# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from braindecode.datasets import MOABBDataset
    from moabb.utils import set_download_dir
    import mne

    from benchmark_utils import windows_data, detect_if_cluster
    from moabb.datasets import (
        AlexMI,
        BNCI2014_001,
        BNCI2014_002,
        BNCI2014_004,
        BNCI2015_001,
        Cho2017,
        Hinss2021,
        Lee2019_MI,
        PhysionetMI,
        Shin2017A,
        Weibo2014,
    )

    # BI2012,
    # BI2013a,
    # BI2014a,
    # BI2014b,
    # BI2015a,
    # BI2015b,
    # BNCI2014_008,
    # BNCI2014_009,
    # BNCI2015_003,
    # BNCI2015_004,
    # CastillosBurstVEP100,
    # CastillosBurstVEP40,
    # CastillosCVEP100,
    # CastillosCVEP40,
    # Cattan2019_PHMD,
    # Cattan2019_VR,
    # EPFLP300,
    # GrosseWentrup2009,
    # Huebner2017,
    # Huebner2018,
    # Kalunga2016,
    # Lee2019_ERP,
    # Lee2019_SSVEP,
    # MAMEM1,
    # MAMEM2,
    # MAMEM3,
    # Nakanishi2015,
    # Ofner2017,
    # Rodrigues2017,
    # Schirrmeister2017,
    # Shin2017B,
    # Sosulski2019,
    # Stieger2021,
    # Thielen2015,
    # Thielen2021,
    # Wang2016,
    # Zhou2016,

    dataset_list = [
        AlexMI,  # 9 Subject,
        BNCI2014_001,  # 10 Subject,
        BNCI2014_002,  # 15 Subject,
        BNCI2014_004,  # 10 subject,
        BNCI2015_001,  # 13 subject,
        Cho2017,  # 53 subject,
        # Hinss2021,  # 16 subject,
        # Lee2019_MI,  # 54 subject,
        # PhysionetMI,  # 109 subject,
        Shin2017A,  # 30 subject,
        Weibo2014,  # 10 subject,
    ]

    dataset_list_str = [obj.__name__ for obj in dataset_list]
    dataset_class = dict(zip(dataset_list_str, dataset_list))


class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "MOABBDatasets"
    parameters = {
        "dataset_name": dataset_list_str,
        "subject_limit": [None],
    }

    def get_data(self):
        """Returns the data to be passed to Objective.set_data.

        Data
        ----
        Dataset: an instance of a braindecode.WindowsDataset
        sfreq: the sampling frequency of the data.
        """
        # import pytest
        # pytest.skip("This dataset is not yet supported")

        running_cluster = detect_if_cluster()
        try:
            if running_cluster is not None:
                if mne.get_config("MNE_DATA") != str(running_cluster):
                    set_download_dir(running_cluster)
        except Exception:
            pass

        if self.dataset_name == "Shin2017A":
            dataset_kwargs = dict(accept=True)
            dataset_obj = dataset_class[self.dataset_name](**dataset_kwargs)
        else:
            dataset_obj = dataset_class[self.dataset_name]()
            dataset_kwargs = None

        dataset_name = self.dataset_name
        paradigm_name = dataset_obj.paradigm
        unit_factor = dataset_obj.unit_factor
        events_labels = dataset_obj.event_id
        subject_list = dataset_obj.subject_list

        if self.subject_limit is None:
            limit = 0
        else:
            limit = self.subject_limit

        if len(subject_list) > limit and len(subject_list) > 5:
            subject_list = subject_list[: self.subject_limit]
        else:
            print(f"Subject list is {subject_list}")

        dataset = MOABBDataset(
            dataset_name,
            subject_ids=subject_list,
            dataset_kwargs=dataset_kwargs,
        )

        dataset, sfreq = windows_data(
            dataset=dataset,
            events_labels=events_labels,
            unit_factor=unit_factor,
            paradigm_name=paradigm_name,
            dataset_name=dataset_name,
        )

        return dict(
            dataset=dataset,
            sfreq=sfreq,
            paradigm_name=paradigm_name,
            dataset_name=dataset_name,
        )
