import {
    useState,
} from "react";

import {
    predictImage,
} from "../services/api";


export function useAnalysis() {

    const [
        result,
        setResult,
    ] = useState(null);

    const [
        loading,
        setLoading,
    ] = useState(false);

    const [
        error,
        setError,
    ] = useState(null);


    async function analyze(file) {

        if (!file) {

            setError(
                "Please select an image."
            );

            return null;
        }


        setLoading(true);

        setError(null);

        setResult(null);


        try {

            const data =
                await predictImage(
                    file
                );

            setResult(data);

            return data;

        } catch (err) {

            const message =
                err instanceof Error
                    ? err.message
                    : "Unable to analyze image.";

            setError(message);

            return null;

        } finally {

            setLoading(false);

        }
    }


    function reset() {

        setResult(null);

        setError(null);

        setLoading(false);
    }


    return {
        result,
        loading,
        error,
        analyze,
        reset,
    };
}