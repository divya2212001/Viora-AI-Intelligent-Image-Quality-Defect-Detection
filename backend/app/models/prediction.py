from datetime import datetime, timezone


def create_prediction_document(
    filename: str,
    prediction: dict,
):

    return {

  
        # IDENTIFICATION
  

        "prediction_id":
            prediction[
                "prediction_id"
            ],

        "filename":
            filename,


  
        # IMAGE
  

        "image_url":
            prediction.get(
                "image_url"
            ),

        "gradcam_url":
            prediction.get(
                "gradcam_url"
            ),


  
        # QUALITY
  

        "quality_score":
            prediction[
                "quality_score"
            ],

        "qmos":
            prediction[
                "qmos"
            ],

        "quality_label":
            prediction[
                "quality_label"
            ],


  
        # DEFECTS
  

        "defects":
            prediction[
                "defects"
            ],


  
        # COMPUTER VISION STATISTICS
  

        "statistics":
            prediction[
                "statistics"
            ],


  
        # RECOMMENDATION
  

        "recommendation":
            prediction[
                "recommendation"
            ],


  
        # TIMESTAMP
  

        "created_at":
            datetime.now(
                timezone.utc
            ),
    }