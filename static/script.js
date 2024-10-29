document.addEventListener('DOMContentLoaded', function () { // adds an event listener to the document object, listening for the 'DOMContentLoaded' event. 
    try {
        const predictionText = document.getElementById('prediction');
        const accuracyText = document.getElementById('accuracy');
        const imageUpload = document.getElementById('imageUpload');
        const imagePreview = document.getElementById('imagePreview');
        const btnCapture = document.getElementById('btn-capture');
        const btnPredict = document.getElementById('btn-predict');
        const detectButtonContainer = document.querySelector('.detectButtonContainer');


        if (!btnCapture || !btnPredict) { 
            throw new Error('Required elements not found.');
        }

        let isImageUploaded = false;

        function resetUI() {
            imagePreview.innerHTML = '';
            document.querySelector('.image-section').style.display = 'none';
            detectButtonContainer.style.display = 'none';
            document.getElementById('predictions').style.display = 'none';
            predictionText.textContent = '';
            accuracyText.textContent = '';
        }
        btnPredict.addEventListener('click', async function () { 
            try {
                
                const formData = new FormData();
                formData.append('file', imageUpload.files[0]); 

                const response = await fetch('/detect', { 
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();
                document.getElementById('predictions').style.display = 'block';

                predictionText.textContent = `${data.result}`;
                accuracyText.textContent = `${data.accuracy}`; 

                detectButtonContainer.style.display = 'none';
            
                
            } catch (error) {
                console.error('Error predicting:', error);
            }
            
        });

        imageUpload.addEventListener('change', function () {
            try {
                resetUI();
        
                const file = this.files[0];
        
                document.getElementById('predictions').style.display = 'none';
        
                if (file) {
                    const reader = new FileReader();
        
                    reader.addEventListener('load', function () {
                        const imageUrl = this.result;
        
                        // Check image quality and brightness
                        checkImageQuality(imageUrl, detectButtonContainer);
                        checkBrightness(imageUrl, detectButtonContainer);
        
                        // Show uploaded image
                        const imgElement = document.createElement('img');
                        imgElement.style.maxWidth = '100%';
                        imgElement.style.maxHeight = '300%';
        
                        imagePreview.innerHTML = '';
                        imagePreview.appendChild(imgElement);
                        document.querySelector('.image-section').style.display = 'block';
        
                        imgElement.src = imageUrl;
        
                        // Show detect button
                        if (canShowDetectButton) {
                            detectButtonContainer.style.display = 'block';
                        } else {
                            detectButtonContainer.style.display = 'none';
                        }
        
                        // Reset prediction text
                        predictionText.textContent = '';
                        accuracyText.textContent = '';
                    });
        
                    reader.readAsDataURL(file);
                } else {
                    // If there is no uploaded file, clear everything
                    imagePreview.innerHTML = '';
                    document.querySelector('.image-section').style.display = 'none';
                    detectButtonContainer.style.display = 'none';
                    document.getElementById('predictions').style.display = 'none';
                    predictionText.textContent = '';
                    accuracyText.textContent = '';
                }
            } catch (error) {
                console.error('Error handling image upload:', error);
            }
        });
        
        // Capture button click event *****//
        btnCapture.addEventListener('click', async function () {
            try {
                resetUI();


                document.getElementById('predictions').style.display = 'none';

                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const videoElement = document.createElement('video'); 
                videoElement.srcObject = stream;
                videoElement.play();

                videoElement.addEventListener('loadedmetadata', () => {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
            
                    canvas.width = videoElement.videoWidth;
                    canvas.height = videoElement.videoHeight;
            
                    document.querySelector('.image-section').style.display = 'block';
            
                    function drawFrame() {
                        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                        requestAnimationFrame(drawFrame);
                    }
                    drawFrame();
            
                    const captureButton = document.createElement('button'); 
                    captureButton.textContent = 'Capture';
                    captureButton.addEventListener('click', async() => { 
                        const brightness = checkCapturedImageBrightness(canvas);
                        if (brightness !== null) {
                            videoElement.pause();
                            captureButton.style.display = 'none';
                            detectButtonContainer.style.display = 'block';
                        }
                    });
                    detectButtonContainer.addEventListener('click', async () => {
                        const imageUrl = canvas.toDataURL('image/png');
                        const blobData = await fetch(imageUrl).then(res => res.blob());
                        const formData = new FormData();
                        formData.append('file', blobData, 'captured_image.png');
                        const response = await fetch('/detect', {
                            method: 'POST',
                            body: formData,
                        
                        });
            
                        const data = await response.json();
    
                        predictionText.textContent = `${data.result}`;
                        accuracyText.textContent = `${data.accuracy}`;
                    });
                    
                    predictionText.textContent = '';
                    accuracyText.textContent = '';
                    imagePreview.innerHTML = '';
                    imagePreview.appendChild(canvas);
                    imagePreview.appendChild(captureButton);
                    
                });
            } catch (error) {
                console.error('Error accessing webcam:', error);
                alert('Error accessing webcam. Please check your camera permissions and ensure you are using a secure connection.');
            }
            resetUI();

        });
        
        imageUpload.addEventListener('change', function () {
            detectButtonContainer.style.display = 'block';
        });

        btnCapture.addEventListener('click', function () {
        });

        btnPredict.addEventListener('click', function () {
            detectButtonContainer.style.display = 'none';
        });

    } catch (error) {
        console.error('Initialization error:', error);
    }
});

let canShowDetectButton = true;


function checkImageQuality(imageUrl, detectButtonContainer) {
    const img = new Image();
    img.onload = function() {
        const width = this.width;
        const height = this.height;

        if (width < 300 || height < 300) {
            showModal("Resloution-popup");
            imagePreview.innerHTML = '';
            detectButtonContainer.style.display = 'none';
            // Hide predictions <ul>
            document.getElementById('predictions').style.display = 'none';
        } 
    };
    img.src = imageUrl;
}
//_________________________________________________________________________
function checkCapturedImageQuality(canvas, detectButtonContainer) {
    const width = canvas.width;
    const height = canvas.height;

    if (width < 690 || height < 490) {
        alert("low resloution")
        imagePreview.innerHTML = '';
        detectButtonContainer.style.display = 'none';
        document.getElementById('predictions').style.display = 'none';
    } 
}

function checkCapturedImageBrightness(canvas, detectButtonContainer) {
    const context = canvas.getContext('2d');
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    const brightness = calculateBrightness(imageData);

    const brightThreshold = 150;
    const darkThreshold = 50; 

    if (brightness > brightThreshold) {
        showModal("Brightness-popup");
        document.getElementById("brightnessMessage").innerText = "The captured image appears very bright. Please ensure proper lighting conditions for accurate analysis.";
        imagePreview.innerHTML = '';
        detectButtonContainer.style.display = 'none';
        document.getElementById('predictions').style.display = 'none';
    } else if (brightness < darkThreshold) {
        showModal("Brightness-popup");
        document.getElementById("brightnessMessage").innerText = "The captured image appears very dark. Please ensure proper lighting conditions for accurate analysis."
        imagePreview.innerHTML = '';
        detectButtonContainer.style.display = 'none';
        document.getElementById('predictions').style.display = 'none';
    }
}
//______________________________________________________________________
function checkBrightness(imageUrl, detectButtonContainer) {
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = function() {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        context.drawImage(img, 0, 0);

        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        const brightness = calculateBrightness(imageData);

        const brightThreshold = 180;
        const darkThreshold = 50; 

        if (brightness > brightThreshold) {
            showModal("Brightness-popup");
            document.getElementById("brightnessMessage").innerText = "The image appears very bright. Please ensure proper lighting conditions for accurate analysis.";
            imagePreview.innerHTML = '';
            detectButtonContainer.style.display = 'none';
            document.getElementById('predictions').style.display = 'none';
        } else if (brightness < darkThreshold) {
            showModal("Brightness-popup");
            document.getElementById("brightnessMessage").innerText = "The image appears very dark. Please ensure proper lighting conditions for accurate analysis."
            imagePreview.innerHTML = '';
            detectButtonContainer.style.display = 'none';
            
            document.getElementById('predictions').style.display = 'none';
        }
    };
    img.src = imageUrl;
}

function calculateBrightness(imageData) {
    const data = imageData.data;
    const length = data.length;
    let sum = 0;
    
    for (let i = 0; i < length; i += 4) {
        
        sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    const averageBrightness = sum / (imageData.width * imageData.height);

    return averageBrightness;
}

function showModal(modalId) {
    var modal = document.getElementById(modalId);
    modal.style.display = "block";
}

function hideModal(modalId) {
    var modal = document.getElementById(modalId);
    modal.style.display = "none";
}


document.addEventListener('DOMContentLoaded', function() {
    const menuIcon = document.getElementById('menu-icon');
    const navbar = document.querySelector('.navbar');
    const activeNav = document.querySelector('.active-nav');

    menuIcon.addEventListener('click', function() {
        navbar.classList.toggle('active');
        activeNav.classList.toggle('active');
    });
});

window.addEventListener('scroll', function () {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('header nav a');

    sections.forEach(sec => {
        const top = window.scrollY;
        const offset = sec.offsetTop - 100;
        const height = sec.offsetHeight;
        const id = sec.getAttribute('id');

        if (top >= offset && top < offset + height) {
            // Remove the "active" class from all navbar links
            navLinks.forEach(links => {
                links.classList.remove('active');
            });

            // Add the "active" to the current section
            const currentNavLink = document.querySelector('header nav a[href="#' + id + '"]');
            if (currentNavLink) {
                currentNavLink.classList.add('active');
            }
        }
    });
});


