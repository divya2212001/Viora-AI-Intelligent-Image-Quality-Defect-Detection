export function getQualityClass(
    qmos
) {

    const score =
        Number(qmos);

    if (score >= 4.0) {
        return "excellent";
    }

    if (score >= 3.5) {
        return "good";
    }

    if (score >= 2.5) {
        return "fair";
    }

    if (score >= 1.5) {
        return "poor";
    }

    return "very-poor";
}


export function getScoreDescription(
    qmos
) {

    const score =
        Number(qmos);

    if (score >= 4.0) {
        return "Excellent image quality with minimal detected issues.";
    }

    if (score >= 3.5) {
        return "Good image quality with only minor issues.";
    }

    if (score >= 2.5) {
        return "Moderate image quality. Some issues may be present.";
    }

    if (score >= 1.5) {
        return "Poor image quality. Several issues may affect the image.";
    }

    return "Very poor image quality. Significant issues were detected.";
}


export function formatPercentage(
    value
) {

    const number =
        Number(value);

    if (Number.isNaN(number)) {
        return "0%";
    }

    return `${(number * 100).toFixed(1)}%`;
}


export function formatScore(
    value,
    decimals = 2
) {

    const number =
        Number(value);

    if (Number.isNaN(number)) {
        return "0";
    }

    return number.toFixed(
        decimals
    );
}


export function formatDate(
    value
) {

    if (!value) {
        return "Unknown";
    }

    const date =
        new Date(value);

    if (Number.isNaN(
        date.getTime()
    )) {
        return String(value);
    }

    return date.toLocaleString();
}


export function getHighestDefect(
    defects = {}
) {

    const entries =
        Object.entries(defects);

    if (entries.length === 0) {
        return null;
    }

    return entries.reduce(
        (highest, current) => {

            return current[1] >
                highest[1]
                ? current
                : highest;

        }
    );
}


export function prettifyName(
    name
) {

    if (!name) {
        return "";
    }

    return String(name)
        .replace(/_/g, " ")
        .replace(/-/g, " ")
        .replace(
            /\b\w/g,
            letter =>
                letter.toUpperCase()
        );
}