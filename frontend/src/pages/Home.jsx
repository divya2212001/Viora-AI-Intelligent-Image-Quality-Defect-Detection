import {
    useState,
} from "react";

import {
    useNavigate,
} from "react-router-dom";

import Header from "../components/Header";
import ImageUploader from "../components/ImageUploader";
import ImagePreview from "../components/ImagePreview";
import LoadingState from "../components/LoadingState";
import ModelInfo from "../components/ModelInfo";

import {
    useAnalysis,
} from "../hooks/useAnalysis";


function Home() {

    const [
        file,
        setFile,
    ] = useState(null);


    const {
        loading,
        error,
        analyze,
    } = useAnalysis();


    const navigate =
        useNavigate();


    async function handleAnalyze() {

        if (!file) {
            return;
        }

        const result =
            await analyze(file);

        if (result) {

            navigate(
                "/results",
                {
                    state: {
                        result,
                    },
                }
            );
        }
    }


    return (

        <div className="app">

            <Header />


            <main className="page">

                <section className="hero">

                    <span className="eyebrow">
                        AI IMAGE QUALITY ANALYSIS
                    </span>

                    <h1>
                        Understand the quality
                        of your image.
                    </h1>

                    <p>
                        Upload an image and our
                        Hybrid CNN + Computer Vision
                        system will estimate perceptual
                        quality and identify visual defects.
                    </p>

                </section>


                <section className="analysis-card">

                    {!file ? (

                        <ImageUploader
                            onFileSelected={
                                setFile
                            }
                            disabled={loading}
                        />

                    ) : (

                        <ImagePreview
                            file={file}
                            onRemove={() =>
                                setFile(null)
                            }
                        />

                    )}


                    {error && (

                        <div className="error-box">
                            {error}
                        </div>

                    )}


                    {loading ? (

                        <LoadingState />

                    ) : (

                        file && (

                            <button
                                className="primary-button analyze-button"
                                onClick={
                                    handleAnalyze
                                }
                            >
                                Analyze Image
                            </button>

                        )

                    )}

                </section>


                <ModelInfo />

            </main>

        </div>
    );
}


export default Home;