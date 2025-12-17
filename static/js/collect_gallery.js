/**
 * Face Data Collector - Gallery JavaScript
 * Logic quản lý gallery cho admin
 */

// ============ DOM Elements ============
const personGrid = document.getElementById('collectPersonGrid');
const imageGrid = document.getElementById('collectImageGrid');
const searchInput = document.getElementById('collectSearchInput');
const sortSelect = document.getElementById('collectSortSelect');
const modal = document.getElementById('collectModal');
const modalImage = document.getElementById('collectModalImage');
const personTitle = document.getElementById('collectPersonTitle');
const backToPersonsBtn = document.getElementById('collectBackToPersons');
const statsTotal = document.getElementById('collectStatsTotalPersons');
const statsImages = document.getElementById('collectStatsTotalImages');
const deleteAllBtn = document.getElementById('collectDeleteAllBtn');

// ============ State ============
let allPersons = [];
let currentPerson = null;
let currentImages = [];
let currentImageIndex = 0;

// ============ Toast Notification ============
function showToast(message, type = 'info') {
    const container = document.getElementById('collectToastContainer');
    const toast = document.createElement('div');
    toast.className = `collect-toast ${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                 type === 'error' ? 'fa-exclamation-circle' :
                 type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle';
    
    toast.innerHTML = `<i class="fas ${icon}"></i> ${message}`;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'collectSlideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============ Modal Functions ============
function showModal(imageSrc, imageIndex = 0) {
    currentImageIndex = imageIndex;
    updateModalImage();
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    modal.style.display = 'none';
    document.body.style.overflow = '';
}

function updateModalImage() {
    if (currentImages.length === 0) return;
    
    const img = currentImages[currentImageIndex];
    modalImage.src = `/collect/output/${currentPerson}/${img.filename}`;
    
    // Update counter
    const counter = document.getElementById('collectModalCounter');
    if (counter) {
        counter.textContent = `${currentImageIndex + 1} / ${currentImages.length}`;
    }
    
    // Show/hide navigation buttons
    const prevBtn = document.getElementById('collectModalPrev');
    const nextBtn = document.getElementById('collectModalNext');
    
    if (prevBtn) {
        prevBtn.style.display = currentImageIndex > 0 ? 'flex' : 'none';
    }
    if (nextBtn) {
        nextBtn.style.display = currentImageIndex < currentImages.length - 1 ? 'flex' : 'none';
    }
}

function showPreviousImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        updateModalImage();
    }
}

function showNextImage() {
    if (currentImageIndex < currentImages.length - 1) {
        currentImageIndex++;
        updateModalImage();
    }
}

// ============ API Functions ============
async function fetchPersons() {
    try {
        const response = await fetch('/collect/api/list_persons');
        const data = await response.json();
        
        if (data.success) {
            allPersons = data.persons;
            updateStats();
            renderPersons(allPersons);
        } else {
            showToast(data.message, 'error');
        }
    } catch (error) {
        console.error('Error fetching persons:', error);
        showToast('Lỗi khi tải danh sách!', 'error');
    }
}

async function fetchImages(folder) {
    try {
        const response = await fetch(`/collect/api/list_images?folder=${encodeURIComponent(folder)}`);
        const data = await response.json();
        
        if (data.success) {
            currentImages = data.images;
            renderImages(currentImages);
        } else {
            showToast(data.message, 'error');
        }
    } catch (error) {
        console.error('Error fetching images:', error);
        showToast('Lỗi khi tải ảnh!', 'error');
    }
}

async function deleteImage(folder, filename) {
    if (!confirm('Bạn có chắc muốn xóa ảnh này?')) {
        return;
    }
    
    try {
        const response = await fetch('/collect/api/delete_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ folder, filename })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(data.message, 'success');
            
            // Update local state
            currentImages = currentImages.filter(img => img.filename !== filename);
            
            if (currentImages.length === 0) {
                // No more images, go back to persons
                backToPersons();
                fetchPersons();
            } else {
                renderImages(currentImages);
                // Update person count
                const person = allPersons.find(p => p.folder === folder);
                if (person) {
                    person.count = currentImages.length;
                    updateStats();
                }
            }
        } else {
            showToast(data.message, 'error');
        }
    } catch (error) {
        console.error('Error deleting image:', error);
        showToast('Lỗi khi xóa ảnh!', 'error');
    }
}

async function deletePerson(folder) {
    const person = allPersons.find(p => p.folder === folder);
    const personName = person ? person.folder : folder;
    
    if (!confirm(`Bạn có chắc muốn xóa TẤT CẢ ảnh của "${personName}"?`)) {
        return;
    }
    
    try {
        const response = await fetch('/collect/api/delete_person', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ folder })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(data.message, 'success');
            
            // Update local state
            allPersons = allPersons.filter(p => p.folder !== folder);
            updateStats();
            renderPersons(filterAndSortPersons());
            
            if (currentPerson === folder) {
                backToPersons();
            }
        } else {
            showToast(data.message, 'error');
        }
    } catch (error) {
        console.error('Error deleting person:', error);
        showToast('Lỗi khi xóa người!', 'error');
    }
}

// ============ Render Functions ============
function renderPersons(persons) {
    if (persons.length === 0) {
        personGrid.innerHTML = `
            <div class="collect-empty-state">
                <i class="fas fa-folder-open"></i>
                <p>Chưa có dữ liệu</p>
            </div>
        `;
        return;
    }
    
    personGrid.innerHTML = persons.map(person => `
        <div class="collect-person-card" data-folder="${person.folder}">
            <div class="collect-person-thumbnail">
                ${person.thumbnail ? 
                    `<img src="/collect/output/${person.folder}/${person.thumbnail}" alt="${person.folder}">` :
                    `<i class="fas fa-user"></i>`
                }
            </div>
            <div class="collect-person-info">
                <h3>${person.folder}</h3>
                <p>${person.count} ảnh</p>
            </div>
            <div class="collect-person-actions">
                <button class="collect-btn collect-btn-small collect-btn-primary" onclick="event.stopPropagation(); viewPerson('${person.folder}')">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="collect-btn collect-btn-small collect-btn-danger" onclick="event.stopPropagation(); deletePerson('${person.folder}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join('');
    
    // Add click event to cards
    document.querySelectorAll('.collect-person-card').forEach(card => {
        card.addEventListener('click', () => {
            viewPerson(card.dataset.folder);
        });
    });
}

function renderImages(images) {
    if (images.length === 0) {
        imageGrid.innerHTML = `
            <div class="collect-empty-state">
                <i class="fas fa-images"></i>
                <p>Không có ảnh</p>
            </div>
        `;
        return;
    }
    
    imageGrid.innerHTML = images.map((img, index) => `
        <div class="collect-image-card">
            <img src="/collect/output/${currentPerson}/${img.filename}" 
                 alt="${img.filename}"
                 onclick="showModal(this.src, ${index})"
                 data-index="${index}">
            <div class="collect-image-overlay">
                <span class="collect-image-name">${img.filename}</span>
                <button class="collect-btn collect-btn-small collect-btn-danger" 
                        onclick="deleteImage('${currentPerson}', '${img.filename}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join('');
}

function updateStats() {
    const totalPersons = allPersons.length;
    const totalImages = allPersons.reduce((sum, p) => sum + p.count, 0);
    
    statsTotal.textContent = totalPersons;
    statsImages.textContent = totalImages;
}

// ============ View Functions ============
function viewPerson(folder) {
    currentPerson = folder;
    personTitle.textContent = folder;
    
    personGrid.style.display = 'none';
    imageGrid.style.display = 'grid';
    backToPersonsBtn.style.display = 'inline-flex';
    
    if (deleteAllBtn) {
        deleteAllBtn.style.display = 'inline-flex';
        deleteAllBtn.onclick = () => deletePerson(folder);
    }
    
    fetchImages(folder);
}

function backToPersons() {
    currentPerson = null;
    personTitle.textContent = 'Danh sách';
    
    personGrid.style.display = 'grid';
    imageGrid.style.display = 'none';
    backToPersonsBtn.style.display = 'none';
    
    if (deleteAllBtn) {
        deleteAllBtn.style.display = 'none';
    }
}

// ============ Filter & Sort ============
function filterAndSortPersons() {
    let result = [...allPersons];
    
    // Filter by search
    const search = searchInput.value.trim().toLowerCase();
    if (search) {
        result = result.filter(p => p.folder.toLowerCase().includes(search));
    }
    
    // Sort
    const sortBy = sortSelect.value;
    switch (sortBy) {
        case 'name-asc':
            result.sort((a, b) => a.folder.localeCompare(b.folder));
            break;
        case 'name-desc':
            result.sort((a, b) => b.folder.localeCompare(a.folder));
            break;
        case 'count-asc':
            result.sort((a, b) => a.count - b.count);
            break;
        case 'count-desc':
            result.sort((a, b) => b.count - a.count);
            break;
    }
    
    return result;
}

// ============ Event Listeners ============
document.addEventListener('DOMContentLoaded', () => {
    fetchPersons();
    
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            renderPersons(filterAndSortPersons());
        });
    }
    
    if (sortSelect) {
        sortSelect.addEventListener('change', () => {
            renderPersons(filterAndSortPersons());
        });
    }
    
    if (backToPersonsBtn) {
        backToPersonsBtn.addEventListener('click', backToPersons);
    }
    
    // Modal close events
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
        
        document.querySelector('.collect-modal-close')?.addEventListener('click', closeModal);
        
        // Navigation buttons
        document.getElementById('collectModalPrev')?.addEventListener('click', (e) => {
            e.stopPropagation();
            showPreviousImage();
        });
        
        document.getElementById('collectModalNext')?.addEventListener('click', (e) => {
            e.stopPropagation();
            showNextImage();
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (modal.style.display === 'flex') {
                if (e.key === 'Escape') {
                    closeModal();
                } else if (e.key === 'ArrowLeft') {
                    showPreviousImage();
                } else if (e.key === 'ArrowRight') {
                    showNextImage();
                }
            }
        });
    }
});
