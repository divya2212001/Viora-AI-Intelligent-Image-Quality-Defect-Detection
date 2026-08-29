import {
    formatScore,
    prettifyName,
} from "../utils/helper";


function Statistics({
    statistics = {},
}) {

    const entries =
        Object.entries(statistics);


    return (

        <section className="panel">

            <div className="section-heading">

                <div>

                    <span className="eyebrow">
                        COMPUTER VISION
                    </span>

                    <h2>
                        Image Statistics
                    </h2>

                </div>

            </div>


            {entries.length === 0 ? (

                <p className="muted">
                    No image statistics available.
                </p>

            ) : (

                <div className="statistics-grid">

                    {entries.map(
                        ([name, value]) => (

                            <div
                                className="stat-card"
                                key={name}
                            >

                                <span>
                                    {prettifyName(
                                        name
                                    )}
                                </span>

                                <strong>
                                    {formatScore(
                                        value,
                                        4
                                    )}
                                </strong>

                            </div>

                        )
                    )}

                </div>

            )}

        </section>
    );
}


export default Statistics;