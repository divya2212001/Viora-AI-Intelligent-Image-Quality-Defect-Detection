import {
    getQualityClass,
    getScoreDescription,
} from "../utils/helper";


function QualityScore({
    qmos,
    qualityScore,
    qualityLabel,
}) {

    const numericQmos =
        Number(qmos) || 0;

    const percentage =
        Math.max(
            0,
            Math.min(
                100,
                Number(qualityScore) ||
                    numericQmos * 20
            )
        );


    const qualityClass =
        getQualityClass(
            numericQmos
        );


    return (

        <section className="quality-card">

            <div className="section-heading">

                <div>

                    <span className="eyebrow">
                        OVERALL QUALITY
                    </span>

                    <h2>
                        Image Quality
                    </h2>

                </div>

                <span
                    className={`quality-badge ${qualityClass}`}
                >
                    {qualityLabel || "Unknown"}
                </span>

            </div>


            <div className="quality-main">

                <div className="score-circle">

                    <div className="score-number">
                        {numericQmos.toFixed(2)}
                    </div>

                    <div className="score-total">
                        / 5.00
                    </div>

                </div>


                <div className="quality-details">

                    <div className="quality-score-large">
                        {percentage.toFixed(1)}
                        <span>/100</span>
                    </div>

                    <p>
                        {getScoreDescription(
                            numericQmos
                        )}
                    </p>

                </div>

            </div>


            <div className="quality-bar">

                <div
                    className="quality-bar-fill"
                    style={{
                        width:
                            `${percentage}%`,
                    }}
                />

            </div>

        </section>
    );
}


export default QualityScore;