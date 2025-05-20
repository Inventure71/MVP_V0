document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('pdfUploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const processingStatus = document.getElementById('processingStatus');
    const statusMessage = document.getElementById('statusMessage');
    const resultContainer = document.getElementById('resultContainer');
    const resultVideo = document.getElementById('resultVideo');
    const downloadBtn = document.getElementById('downloadBtn');
    
    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const files = document.getElementById('pdfFiles').files;
        const customInstructions = document.getElementById('customInstructions').value;
        const ttsEngine = document.getElementById('ttsEngine').value;
        
        if (files.length === 0) {
            alert('Please select at least one PDF file to upload.');
            return;
        }
        
        // Show processing status and hide form
        form.parentElement.classList.add('hidden');
        processingStatus.classList.remove('hidden');
        statusMessage.textContent = 'Uploading files...';
        
        // Create FormData object to send files
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        
        // Add custom instructions if provided
        if (customInstructions.trim()) {
            formData.append('custom_instructions', customInstructions.trim());
        }
        
        // Add selected TTS engine
        formData.append('tts_engine', ttsEngine);
        
        try {
            // Send request to backend
            statusMessage.textContent = 'Processing your PDF files...';
            
            const response = await fetch('/process-pdfs/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status} - ${response.statusText}`);
            }
            
            // Get blob data (video file)
            const blob = await response.blob();
            const videoUrl = URL.createObjectURL(blob);
            
            // Hide processing status and show result
            processingStatus.classList.add('hidden');
            resultContainer.classList.remove('hidden');
            
            // Set video source
            resultVideo.src = videoUrl;
            
            // Set up download button
            downloadBtn.onclick = () => {
                const a = document.createElement('a');
                a.href = videoUrl;
                a.download = 'lesson_video.mp4';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            };
            
        } catch (error) {
            console.error('Error:', error);
            statusMessage.textContent = `Error: ${error.message}`;
            statusMessage.style.color = 'red';
            
            // Add a retry button
            const retryBtn = document.createElement('button');
            retryBtn.textContent = 'Try Again';
            retryBtn.onclick = () => {
                processingStatus.classList.add('hidden');
                form.parentElement.classList.remove('hidden');
            };
            processingStatus.appendChild(retryBtn);
        }
    });
}); 