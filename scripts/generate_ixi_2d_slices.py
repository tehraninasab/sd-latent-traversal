import nibabel as nib
import numpy as np
import os
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk

class MRIViewer:
    def __init__(self, nii_file):
        """
        Initialize MRI viewer with a NIfTI file.
        """
        self.img = nib.load(nii_file)
        self.data = self.img.get_fdata()
        self.shape = self.data.shape
        #print(f"Image shape: {self.shape}")

    def get_slice(self, slice_num, view='coronal'):
        """
        Get a specific slice in the specified orientation.
        """
        if view == 'sagittal':
            slice_img = self.data[slice_num, :, :]
        elif view == 'coronal':
            slice_img = self.data[:, slice_num, :]
        elif view == 'axial':
            slice_img = self.data[:, :, slice_num]
        else:
            raise ValueError("View must be one of: sagittal, coronal, axial")
            
        # Rotate for correct orientation
        if view == 'coronal':
            slice_img = np.rot90(slice_img, k=2)
        elif view == 'axial':
            slice_img = np.rot90(slice_img, k=1)
            
        return slice_img
    
    def find_eye_end_slice(self, view='coronal'):
        """
        Find the slice number where the eyes end (top of orbital cavity).
        
        Parameters:
        view (str): 'coronal' or 'axial' view
        """
        if view not in ['coronal', 'axial']:
            raise ValueError("View must be either 'coronal' or 'axial'")
            
        if view == 'coronal':
            return self._find_eye_end_slice_coronal()
        else:
            return self._find_eye_end_slice_axial()

    def _find_eye_end_slice_coronal(self):
        """
        Find eye end slice in coronal view.
        """
        # Get middle sagittal position for reference
        mid_sagittal = self.data.shape[0] // 2
        
        # Calculate intensity profile for each slice
        slice_profiles = []
        for i in range(self.data.shape[1]):
            slice_img = self.data[:, i, :]
            
            # Look at regions where eyes typically appear
            left_eye_region = slice_img[mid_sagittal-40:mid_sagittal-10, :]
            right_eye_region = slice_img[mid_sagittal+10:mid_sagittal+40, :]
            
            # Calculate metrics for both eye regions
            left_mean = np.mean(left_eye_region)
            right_mean = np.mean(right_eye_region)
            left_std = np.std(left_eye_region)
            right_std = np.std(right_eye_region)
            
            # Look for symmetry between left and right regions
            symmetry_score = 1 - abs(left_mean - right_mean) / max(left_mean, right_mean)
            
            # Combined eye score considering both regions
            eye_score = (left_std + right_std) * symmetry_score * (2 - (left_mean + right_mean) / np.max(slice_img))
            
            slice_profiles.append(eye_score)
            
        return self._process_eye_scores(slice_profiles)

    def _find_eye_end_slice_axial(self):
        """
        Find eye end slice in axial view.
        """
        # Calculate intensity profile for each slice
        slice_profiles = []
        for i in range(self.data.shape[2]):
            slice_img = self.data[:, :, i]
            
            # Look at central region where eyes typically appear
            center_x = slice_img.shape[0] // 2
            center_y = slice_img.shape[1] // 2
            
            # Define eye regions in axial view
            region_size = 15
            eye_region = slice_img[center_x-region_size:center_x+region_size, 
                                 center_y-region_size:center_y+region_size]
            
            # Calculate metrics
            mean_intensity = np.mean(eye_region)
            std_intensity = np.std(eye_region)
            
            # Score based on intensity variation and mean
            eye_score = std_intensity * (1 - mean_intensity/np.max(slice_img))
            
            slice_profiles.append(eye_score)
            
        return self._process_eye_scores(slice_profiles)

    def _process_eye_scores(self, slice_profiles):
        """
        Process eye scores to find end slice.
        """
        eye_scores = np.array(slice_profiles)
        
        # Smooth the scores
        smoothed_scores = gaussian_filter1d(eye_scores, sigma=2)
        
        # Calculate gradient
        gradient = np.gradient(smoothed_scores)
        
        # Consider middle third where eyes typically appear
        shape = len(smoothed_scores)
        start_idx = shape // 3
        end_idx = (2 * shape) // 3
        
        # Find where eye pattern ends
        eye_region = smoothed_scores[start_idx:end_idx]
        eye_gradients = gradient[start_idx:end_idx]
        
        # Find peak of eye scores
        peak_idx = start_idx + np.argmax(eye_region)
        
        # Look for strong negative gradient after peak
        post_peak_gradients = eye_gradients[np.argmax(eye_region):]
        end_idx = np.argmin(post_peak_gradients)
        
        eye_end_slice = peak_idx + end_idx
        
        print(f"Detected eye end at slice {eye_end_slice}")
        return eye_end_slice
    
    def calculate_axial_brain_area(self, slice_img):
        """
        Calculate the brain area in an axial slice using advanced image processing.
        
        Parameters:
        slice_img (numpy.ndarray): 2D array representing the axial slice
        
        Returns:
        float: Area of the brain in the slice
        dict: Additional metrics including centroid and symmetry
        """
        # Normalize image
        normalized = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
        
        # Apply Otsu's thresholding
        thresh = threshold_otsu(normalized)
        binary = normalized > thresh
        
        # Fill holes and clean up the mask
        filled = binary_fill_holes(binary)
        cleaned = binary_closing(filled, disk(3))
        
        # Calculate area
        area = np.sum(cleaned)
        
        # Calculate centroid
        y_indices, x_indices = np.nonzero(cleaned)
        if len(x_indices) > 0 and len(y_indices) > 0:
            centroid_x = np.mean(x_indices)
            centroid_y = np.mean(y_indices)
        else:
            centroid_x = centroid_y = 0
            
        # Calculate symmetry score
        width = cleaned.shape[1]
        left_half = cleaned[:, :width//2]
        right_half = cleaned[:, width//2:]
        right_half_flipped = np.fliplr(right_half)
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry_score = np.sum(left_half[:, :min_width] == right_half_flipped[:, :min_width]) / (min_width * cleaned.shape[0])
        
        metrics = {
            'area': area,
            'centroid': (centroid_x, centroid_y),
            'symmetry': symmetry_score
        }
        
        return area, metrics

    def find_largest_axial_slices(self, num_slices=5, min_spacing=3):
        """
        Find axial slices with the largest brain area, ensuring minimum spacing between slices.
        
        Parameters:
        num_slices (int): Number of slices to find
        min_spacing (int): Minimum number of slices between selected slices
        
        Returns:
        list: List of (slice_number, area, metrics) tuples for best slices
        """
        # Find eye end slice as lower boundary
        eye_end = self.find_eye_end_slice('axial')
        
        # Analyze all slices above eye level
        slice_data = []
        for i in range(0, self.data.shape[2]):
            slice_img = self.get_slice(i, 'axial')
            area, metrics = self.calculate_axial_brain_area(slice_img)
            
            # Only consider slices with good symmetry and centered brain
            if metrics['symmetry'] > 0.8:  # High symmetry threshold
                slice_data.append((i, area, metrics))
        
        # Sort by area in descending order
        sorted_slices = sorted(slice_data, key=lambda x: x[1], reverse=True)
        
        # Select slices with minimum spacing
        selected_slices = []
        for slice_info in sorted_slices:
            slice_num = slice_info[0]
            if not selected_slices or all(abs(slice_num - s[0]) >= min_spacing for s in selected_slices):
                selected_slices.append(slice_info)
                if len(selected_slices) == num_slices:
                    break
        
        # Sort by slice number for display
        return sorted(selected_slices, key=lambda x: x[0])

    def display_largest_axial_slices(self, num_slices=5, min_spacing=3):
        """
        Display axial slices with the largest brain areas.
        
        Parameters:
        num_slices (int): Number of slices to display
        min_spacing (int): Minimum spacing between selected slices
        
        Returns:
        list: List of selected slice numbers
        """
        best_slices = self.find_largest_axial_slices(num_slices, min_spacing)
        
        # Create figure
        fig, axs = plt.subplots(1, len(best_slices), figsize=(4*len(best_slices), 5))
        if num_slices == 1:
            axs = [axs]
        
        # Display each slice
        for i, (slice_num, area, metrics) in enumerate(best_slices):
            slice_img = self.get_slice(slice_num, 'axial')
            
            axs[i].imshow(slice_img, cmap='gray')
            axs[i].axis('off')
            title = f'Slice {slice_num}\nArea: {int(area)}\nSymmetry: {metrics["symmetry"]:.2f}'
            axs[i].set_title(title)
        
        plt.suptitle('Largest Brain Area Axial Slices', y=1.05)
        plt.tight_layout()
        plt.show()
        
        return [s[0] for s in best_slices]

    def save_largest_axial_slices(self, num_slices=5, min_spacing=2, name="dummy"):
        """
        Display axial slices with the largest brain areas.
        
        Parameters:
        num_slices (int): Number of slices to display
        min_spacing (int): Minimum spacing between selected slices
        
        Returns:
        list: List of selected slice numbers
        """
        best_slices = self.find_largest_axial_slices(num_slices, min_spacing)
        
        # Display each slice
        for i, (slice_num, area, metrics) in enumerate(best_slices):
            slice_img = self.get_slice(slice_num, 'axial')
            #save the image
            plt.imsave(f"/usr/local/faststorage/datasets/IXI/slice-T2/{name}_{slice_num}.png", slice_img, cmap='gray')
                    
        return [s[0] for s in best_slices]
# Example usage
if __name__ == "__main__":
    path = '/usr/local/faststorage/datasets/IXI/IXI-T2'
    image_num = 1
    #loop over all the *.nii.gz files in the directory
    for file in os.listdir(path):
        if file.endswith(".nii.gz"):
            print('Processing image:', image_num)
            nii_file = os.path.join(path, file)
            viewer = MRIViewer(nii_file)
            name = nii_file.split("/")[-1].split(".")[0]
            selected_slices = viewer.save_largest_axial_slices(num_slices=15, min_spacing=2, name=name)
            image_num += 1
    