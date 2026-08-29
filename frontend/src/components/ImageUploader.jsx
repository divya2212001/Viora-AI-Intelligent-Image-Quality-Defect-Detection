import {
    useRef,
    useState,
} from "react";


function ImageUploader({
    onFileSelected,
    disabled = false,
}) {

    const inputRef =
        useRef(null);

    const [
        dragging,
        setDragging,
    ] = useState(false);


    function selectFile(file) {

        if (!file) {
            return;
        }

        if (!file.type.startsWith("image/")) {

            alert(
                "Please select a valid image file."
            );

            return;
        }

        onFileSelected(file);
    }


    function handleInput(event) {

        const file =
            event.target.files?.[0];

        selectFile(file);
    }


    function handleDrop(event) {

        event.preventDefault();

        setDragging(false);

        if (disabled) {
            return;
        }

        const file =
            event.dataTransfer.files?.[0];

        selectFile(file);
    }


    return (

        <div
            className={
                dragging
                    ? "upload-box dragging"
                    : "upload-box"
            }

            onDragOver={(event) => {

                event.preventDefault();

                if (!disabled) {
                    setDragging(true);
                }

            }}

            onDragLeave={() =>
                setDragging(false)
            }

            onDrop={handleDrop}

            onClick={() => {

                if (!disabled) {
                    inputRef.current?.click();
                }

            }}
        >

            <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleInput}
                disabled={disabled}
                hidden
            />


            <div className="upload-icon">
                ↑
            </div>


            <h3>
                Upload an image
            </h3>


            <p>
                Drag & drop your image here
                or click to browse
            </p>


            <span className="upload-format">
                JPG · PNG · WEBP
            </span>

        </div>
    );
}


export default ImageUploader;