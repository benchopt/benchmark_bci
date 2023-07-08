from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    DummyClassifier()
)

# this is what will be loaded
PIPELINE = {
    "name": "DUMMY",
    "paradigms": ["LeftRightImagery"],
    "pipeline": pipe,
}
