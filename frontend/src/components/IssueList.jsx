import {
    formatPercentage,
} from "../utils/helper";

const defectLabels = {
    blur: "Blur",
    underexposure: "Underexposure",
    overexposure: "Overexposure",
    noise: "Noise",
    corruption: "Corruption",
    defect: "Potential Visual Defect",
};

function severity(probability) {
    if (probability >= 0.7) return "High";
    if (probability >= 0.4) return "Medium";
    return "Low";
}


function IssueList({
    defects = {},
}) {

    const entries =
        Object.entries(defects)
            .sort(
                ([, a], [, b]) =>
                    Number(b) - Number(a)
            );


    if (entries.length === 0) {

        return (

            <section className="panel">

                <div className="section-heading">

                    <div>

                        <span className="eyebrow">
                            DEFECT ANALYSIS
                        </span>

                        <h2>
                            Detected Issues
                        </h2>

                    </div>

                </div>

                <p className="muted">
                    No defect information available.
                </p>

            </section>
        );
    }


    return (

        <section className="panel">

            <div className="section-heading">

                <div>

                    <span className="eyebrow">
                        DEFECT ANALYSIS
                    </span>

                    <h2>
                        Detected Issues
                    </h2>

                </div>

            </div>


            <div className="issue-list">

                {entries.map(
                    ([name, value]) => {

                        const probability =
                            Math.max(
                                0,
                                Math.min(
                                    1,
                                    Number(value) || 0
                                )
                            );


                        return (

                            <div
                                className="issue-row"
                                key={name}
                            >

                                <div className="issue-top">

                                    <span className="issue-name">
                                        {defectLabels[name] || name}
                                    </span>

                                    <span className="issue-value">
                                        {formatPercentage(
                                            probability
                                        )}
                                        {` · ${severity(probability)} severity`}
                                    </span>

                                </div>


                                <div className="issue-bar">

                                    <div
                                        className="issue-bar-fill"
                                        style={{
                                            width:
                                                `${probability * 100}%`,
                                        }}
                                    />

                                </div>

                            </div>

                        );

                    }
                )}

            </div>

        </section>
    );
}


export default IssueList;
