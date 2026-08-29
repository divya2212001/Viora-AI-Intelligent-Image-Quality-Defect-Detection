import {
    useEffect,
    useState,
} from "react";

import {
    getModelInfo,
} from "../services/api";


function ModelInfo() {

    const [
        model,
        setModel,
    ] = useState(null);


    useEffect(() => {

        getModelInfo()
            .then(setModel)
            .catch(() => {
                setModel(null);
            });

    }, []);


    return (

        <section className="model-info">

            <div>

                <span className="eyebrow">
                    AI MODEL
                </span>

                <h3>
                    {model?.name ||
                        "ImageQualityNet"}
                </h3>

            </div>


            <div className="model-details">

                <span>
                    Version
                </span>

                <strong>
                    {model?.version || "1.0.0"}
                </strong>

            </div>


            <div className="model-details">

                <span>
                    Architecture
                </span>

                <strong>
                    {model?.architecture ||
                        "Hybrid CNN + Computer Vision"}
                </strong>

            </div>

        </section>
    );
}


export default ModelInfo;
