document.getElementById('image-input').addEventListener('change', function(event) {
    const imageArea = document.getElementById('image-area');
    const uploadedImage = document.getElementById('uploaded-image');
    const file = event.target.files[0];

    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();

        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = 'block';
            imageArea.querySelector('span').style.display = 'none'; // Hide the "No Image Uploaded" text
        };

        reader.readAsDataURL(file); // Read the image file as a data URL
    } else {
        alert("Please upload a valid image file.");
    }
});


function showLoading(event) {
    // Prevent the button from submitting immediately
    event.preventDefault();

    const button = document.getElementById('main_button');
    button.innerHTML = "Loading...";  // Change the button text to 'Loading...'
    button.disabled = true;  // Disable the button to prevent multiple clicks

    // Select the form by its class name
    const form = document.querySelector('.main_form'); // Use querySelector to select by class

    // Submit the form after a short delay to show loading text
    setTimeout(function() {
        form.submit(); // Now trigger the form submission
    }, 300);  // Delay of 300ms, you can adjust this as needed
}
