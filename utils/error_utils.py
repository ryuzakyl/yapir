# general errors
SUCCESS = 0     # like C or C++ (success is 0)
UNKNOWN_FAIL = 1
NOT_IMPLEMENTED_FEATURE = 2

#image errors
INVALID_IMAGE = 3
LOAD_IMAGE_ERROR = 4

# iris segmentation errors
PUPIL_DETECTION_FAILED = 5
IRIS_DETECTION_FAILED = 6
EYELIDS_DETECTION_FAILED = 7
WRONG_IMAGE_FORMAT = 8

# normalization errors
SEGMENTATION_REQUIRED = 9

# iris encoding errors
RESOLUTION_ERROR = 10
UNKNOWN_ENCODING_METHOD = 11


#filling message errors
error_messages = \
    {
        # general errors
        SUCCESS: "Operation completed successfully.",
        UNKNOWN_FAIL: "Unknown error.",
        NOT_IMPLEMENTED_FEATURE: "Feature not available at the moment.",

        # image errors
        INVALID_IMAGE:"Image must be valid.",
        LOAD_IMAGE_ERROR: "There was a problem loading the image.",

        # segmentation errors
        PUPIL_DETECTION_FAILED: "Pupil detection failed.",
        IRIS_DETECTION_FAILED: "Iris detection failed.",
        EYELIDS_DETECTION_FAILED: "Eyelids detection failed.",
        WRONG_IMAGE_FORMAT: "Wrong image format.",

        # normalization errors
        SEGMENTATION_REQUIRED: "Iris segmentation required",

        # encoding errors
        RESOLUTION_ERROR: "Angular resolution * Radial resolution = the amount of encoded pixels.",
        UNKNOWN_ENCODING_METHOD: "Unknown encoding method."
    }