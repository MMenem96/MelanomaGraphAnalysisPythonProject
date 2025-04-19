// JavaScript for Melanoma Detection Application

document.addEventListener('DOMContentLoaded', function() {
    
    // Image preview functionality
    const imageInput = document.getElementById('imageInput');
    const previewImg = document.getElementById('previewImg');
    const imagePreview = document.getElementById('imagePreview');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const imageUploadForm = document.getElementById('imageUploadForm');
    
    if (imageInput) {
        imageInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    imagePreview.classList.remove('d-none');
                };
                
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
    
    // Show loading spinner during form submission
    if (imageUploadForm) {
        imageUploadForm.addEventListener('submit', function() {
            if (imageInput.files && imageInput.files[0]) {
                loadingSpinner.classList.remove('d-none');
                
                // Optional: hide other elements if desired
                if (imagePreview) {
                    imagePreview.classList.add('d-none');
                }
            }
        });
    }
    
    // Tooltips initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Image result enlargement on click
    const resultImages = document.querySelectorAll('.img-thumbnail');
    resultImages.forEach(img => {
        img.addEventListener('click', function() {
            // Create modal to show larger image
            const modal = document.createElement('div');
            modal.classList.add('modal', 'fade');
            modal.id = 'imageModal';
            modal.setAttribute('tabindex', '-1');
            modal.setAttribute('aria-hidden', 'true');
            
            modal.innerHTML = `
                <div class="modal-dialog modal-lg modal-dialog-centered">
                    <div class="modal-content bg-dark">
                        <div class="modal-header">
                            <h5 class="modal-title">${this.alt}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body text-center">
                            <img src="${this.src}" class="img-fluid" alt="${this.alt}">
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            const imageModal = new bootstrap.Modal(document.getElementById('imageModal'));
            imageModal.show();
            
            // Remove modal from DOM after it's hidden
            document.getElementById('imageModal').addEventListener('hidden.bs.modal', function() {
                document.body.removeChild(modal);
            });
        });
        
        // Change cursor on hover
        img.style.cursor = 'pointer';
    });
});
