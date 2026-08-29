function ImagePreview({
    file,
    onRemove,
}) {

    if (!file) {
        return null;
    }

    const previewUrl =
        URL.createObjectURL(file);


    return (

        <div className="image-preview-card">

            <div className="image-preview-header">

                <div>

                    <h3>
                        Selected Image
                    </h3>

                    <p>
                        {file.name}
                    </p>

                </div>


                {onRemove && (

                    <button
                        className="secondary-button"
                        onClick={onRemove}
                    >
                        Remove
                    </button>

                )}

            </div>


            <img
                src={previewUrl}
                alt="Selected"
                className="preview-image"
                onLoad={() =>
                    URL.revokeObjectURL(
                        previewUrl
                    )
                }
            />

        </div>
    );
}


export default ImagePreview;