from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from moabb.pipelines.utils import parse_pipelines_from_directory


def parser_pipelines(dir_path="pipelines", filtering_paradigm=None):
    """
    Read pipelines from a directory and return a list of pipelines.
    """
    if filtering_paradigm is None:
        filtering_paradigm = ["LeftRightImagery", "MotorImagery"]

    pipeline_configs = parse_pipelines_from_directory(dir_path)
    pipelines = {}
    for pipeline in pipeline_configs:

        if any(par in filtering_paradigm for par in pipeline["paradigms"]):
            pipelines.update({pipeline["name"]: pipeline["pipeline"]})
    return pipelines
