from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from braindecode.datasets import MOABBDataset
    from moabb.utils import set_download_dir
    from benchmark_utils import windows_data, detect_if_cluster
    from moabb.datasets import (
        AlexMI,
        BI2012,
        BI2013a,
        BI2014a,
        BI2014b,
        BI2015a,
        BI2015b,
        BNCI2014_001,
        BNCI2014_002,
        BNCI2014_004,
        BNCI2014_008,
        BNCI2014_009,
        BNCI2015_001,
        BNCI2015_003,
        BNCI2015_004,
        CastillosBurstVEP100,
        CastillosBurstVEP40,
        CastillosCVEP100,
        CastillosCVEP40,
        Cattan2019_PHMD,
        Cattan2019_VR,
        Cho2017,
        EPFLP300,
        GrosseWentrup2009,
        Hinss2021,
        Huebner2017,
        Huebner2018,
        Kalunga2016,
        Lee2019_ERP,
        Lee2019_MI,
        Lee2019_SSVEP,
        MAMEM1,
        MAMEM2,
        MAMEM3,
        Nakanishi2015,
        Ofner2017,
        PhysionetMI,
        Rodrigues2017,
        Schirrmeister2017,
        Shin2017A,
        Shin2017B,
        Sosulski2019,
        Stieger2021,
        Thielen2015,
        Thielen2021,
        Wang2016,
        Weibo2014,
        Zhou2016,
    )


class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "MOABBDatasets"
    parameters = {
        "dataset_class": [
            AlexMI,
            BI2012,
            BI2013a,
            BI2014a,
            BI2014b,
            BI2015a,
            BI2015b,
            BNCI2014_001,
            BNCI2014_002,
            BNCI2014_004,
            BNCI2014_008,
            BNCI2014_009,
            BNCI2015_001,
            BNCI2015_003,
            BNCI2015_004,
            CastillosBurstVEP100,
            CastillosBurstVEP40,
            CastillosCVEP100,
            CastillosCVEP40,
            Cattan2019_PHMD,
            Cattan2019_VR,
            Cho2017,
            EPFLP300,
            GrosseWentrup2009,
            Hinss2021,
            Huebner2017,
            Huebner2018,
            Kalunga2016,
            Lee2019_ERP,
            Lee2019_MI,
            Lee2019_SSVEP,
            MAMEM1,
            MAMEM2,
            MAMEM3,
            Nakanishi2015,
            Ofner2017,
            PhysionetMI,
            Rodrigues2017,
            Schirrmeister2017,
            Shin2017A,
            Shin2017B,
            Sosulski2019,
            Stieger2021,
            Thielen2015,
            Thielen2021,
            Wang2016,
            Weibo2014,
            Zhou2016,
        ]
    }

    def get_data(self):
        """Returns the data to be passed to Objective.set_data.

        Data
        ----
        Dataset: an instance of a braindecode.WindowsDataset
        sfreq: the sampling frequency of the data.
        """
        import pytest
        pytest.skip("This dataset is not yet supported")

        running_cluster = detect_if_cluster()
        if running_cluster is not None:
            set_download_dir(running_cluster)

        dataset_obj = self.dataset_class()
        paradigm_name = dataset_obj.paradigm
        dataset_name = dataset_obj.__class__.__name__
        # interval = dataset_obj.interval
        unit_factor = dataset_obj.unit_factor
        events_labels = dataset_obj.event_id

        dataset = MOABBDataset(dataset_name, subject_ids=None)

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
