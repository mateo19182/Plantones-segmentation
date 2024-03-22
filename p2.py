import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
import cv2
import os
from matplotlib import patches
from skimage import img_as_float 
from skimage import img_as_ubyte
from sklearn import metrics as SkMetrics
from skimage import io
from skimage import util
from scipy import ndimage as ndi
from skimage import color
from skimage import metrics
from skimage import data
from skimage import measure
from skimage import draw
from skimage import transform
from skimage import exposure
from skimage import filters
from skimage import feature
from skimage import morphology
from skimage import segmentation
#from plantcv import plantcv as pcv
#from plantcv.parallel import WorkflowInputs
#from sklearn.cluster import KMeans

########################################################

def showImgs (img1, img2, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img1) #, cmap='gray'
    axes[0].set_title('Original')

    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Image')

    for ax in axes:
        ax.axis('off')
    plt.show()

def showImgsgray (img1, img2, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img1, cmap='gray') #
    axes[0].set_title('Original')

    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title('Image')

    for ax in axes:
        ax.axis('off')
    plt.show()

def hsv2bin(img):
    outImg = np.zeros((img.shape[0], img.shape[1]))    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,2] > 0:
                outImg[i,j] = 1
            else:
                outImg[i,j] = 0
    return outImg

def bin(img):
    bool_img = img > 0.5
    return bool_img

def green2bin(img):
    green_channel = img[:, :, 1]
    binary_image = np.where(green_channel > 0.3, 1, 0)
    return binary_image

def maximize_green_contrast(image):
    green_channel = image[:, :, 1] 
    green_eq = exposure.equalize_adapthist(green_channel)
    image_eq = np.copy(image)
    image_eq[:, :, 1] = green_eq
    return image_eq

def pad_img(img, original, region_bbox, of):
    minr, minc, maxr, maxc = region_bbox
    pad_before_row = minr-of
    pad_after_row = original.shape[0] - maxr - of
    pad_before_col = minc-of
    pad_after_col = original.shape[1] - maxc - of
    
    pad_before_row = np.clip(pad_before_row, 0, None)
    pad_before_col = np.clip(pad_before_col, 0, None)
    pad_after_row = np.clip(pad_after_row, 0, None)
    pad_after_col = np.clip(pad_after_col, 0, None)
    img = np.pad(img, ((pad_before_row, pad_after_row), (pad_before_col, pad_after_col)), 'constant')
    return img

def pad_image_center(img, target_img):
    pad_height = target_img.shape[0] - img.shape[0]
    pad_width = target_img.shape[1] - img.shape[1]

    pad_height1, pad_height2 = pad_height // 2, pad_height // 2 + pad_height % 2
    pad_width1, pad_width2 = pad_width // 2, pad_width // 2 + pad_width % 2

    padded_img = np.pad(img, ((pad_height1, pad_height2), (pad_width1, pad_width2)))

    return padded_img

def check_overflow(image):
    sum = 0
    #print(image)
    sum += np.sum(image[0, :] == 1)
    sum += np.sum(image[-1, :] == 1)
    sum += np.sum(image[:, 0] == 1)
    sum += np.sum(image[:, -1] == 1)
    return sum

def isolate_green(image):
    hsv_image = color.rgb2hsv(image)
    lower_green = np.array([0.1, 0.35, 0.25])  
    upper_green = np.array([0.5, 1.0, 1]) 
    green_mask = np.all((hsv_image >= lower_green) & (hsv_image <= upper_green), axis=-1)
    green_parts = image.copy()
    green_parts[~green_mask] = 0
    outImg = color.hsv2rgb(green_parts)
    return outImg

def PCA_color_aug(image):
    # análisis PCA en el espacio de color RGB, https://aparico.github.io/
    orig_img = np.copy(image.astype(np.float64))
    i_a = np.asarray(image)
    i_a = i_a / 255.0
    img_rs = i_a.reshape(-1, 3)
    img_centered = img_rs - np.mean(img_rs, axis=0)
    img_cov = np.cov(img_centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]
    m1 = np.column_stack((eig_vecs))
    m2 = np.zeros((3, 1))
    alpha = np.random.normal(0, 0.1) #intensidad de la modificación, random?
    m2[:, 0] = alpha * eig_vals[:]
    add_vect = np.matrix(m1) * np.matrix(m2)
    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]
    orig_img = np.clip(orig_img, 0.0, 255.0)
    orig_img = orig_img.astype(np.uint8)
    showImgs(image, orig_img, "og")
    return orig_img

def overlap(region1, region2):
    bbox1 = region1.bbox
    bbox2 = region2.bbox

    overlap_x = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    overlap_y = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))

    area1 = region1.area
    area2 = region2.area
    total_overlap = overlap_x * overlap_y
    overlap_ratio = total_overlap / max(area1, area2)
    if overlap_ratio > 0.01:
        print(overlap_ratio)
    return overlap_ratio > 0.1

def remove_overlap(regions):
    remove_mask = np.zeros_like(regions, dtype=bool)
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions[i + 1:]):
            j += i + 1
            if overlap(region1, region2):
                #print("removing region", i, j)
                if region1.area < region2.area:
                    remove_mask[regions == i + 1] = True
                else:
                    remove_mask[regions == j + 1] = True
    clean_regions = np.copy(regions)
    clean_regions[remove_mask] = 0
    return clean_regions

def contrast_hsv(image):
    #Equalize only V channel 
    K = image.copy()
    Khsv = cv2.cvtColor(K, cv2.COLOR_RGB2HSV)
    Khsv[:,:,2] = cv2.equalizeHist(K[:,:,2])
    Knew = cv2.cvtColor(Khsv, cv2.COLOR_HSV2RGB)
    plt.figure(figsize=(6,9))
    plt.imshow(cv2.cvtColor(image))
    plt.title('original')
    plt.show()
    plt.figure(figsize=(6,9))
    plt.imshow(cv2.cvtColor(Knew))
    plt.title('V ecualizado')
    plt.show()

def hough(img):
    result = transform.hough_ellipse(img, accuracy=2, threshold=30, min_size=20, max_size=None)
    result.sort(order='accumulator')
    if result.size > 0:
        best = list(result[-1])
        yc, xc, a, b = (int(round(x)) for x in best[1:5])
        orientation = best[5]
        cy, cx = draw.ellipse_perimeter(yc, xc, a, b, orientation)
        print(yc, xc, a, b, orientation)
        #img[cy, cx] = (0, 0, 255)
        img = color.gray2rgb(img_as_ubyte(img))
        #img[cy, cx] = (250, 0, 0)
        fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),sharex=True, sharey=True)
        ax1.set_title('Original picture')
        ax1.imshow(image)
        ax2.set_title('Edge (white) and result (red)')
        ax2.imshow(img)
        plt.show()    
    else:
        print("No shapes were found.")

def water(result_image_bool, min_dist):
    #algoritmo watershed
    distance = ndi.distance_transform_edt(result_image_bool)
    coords = feature.peak_local_max(distance, labels=result_image_bool, min_distance=min_dist) #, threshold_rel=0.6
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=result_image_bool) #, compactness=0.2
    unique_labels, counts = np.unique(labels, return_counts=True)
    #print(f'regions: {len(unique_labels) - 1}')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_truth) #, cmap='gray'
    axes[0].set_title('truth')
    axes[1].imshow(labels, cmap=plt.cm.nipy_spectral)
    axes[1].set_title('watershed')    
    plt.show()
    return labels

def eval(true, pred):
    #obtención de métricas
    ssim=metrics.structural_similarity(image_truth_bool, result_image_bool)
    #confusion_matrix = SkMetrics.confusion_matrix(true, pred)
    precision = SkMetrics.precision_score(true, pred)
    f1 = SkMetrics.f1_score(true, pred)
    recall = SkMetrics.recall_score(true, pred)
    accuracy = SkMetrics.accuracy_score(true, pred)

    #ROC curve
    fpr, tpr, thresholds = SkMetrics.roc_curve(true, pred)
    roc_auc = SkMetrics.auc(fpr, tpr)
    #plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
    #plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.legend(loc="lower right")
    #plt.show()
    return ssim, precision, f1, recall, accuracy

colors = [
    [255, 0, 0],     # Red
    [0, 255, 0],     # Green
    [0, 0, 255],     # Blue
    [255, 255, 0],   # Yellow
    [0, 255, 255],   # Cyan
    [255, 0, 255],   # Magenta
    [192, 192, 192], # Silver
    [128, 128, 128], # Gray
    [128, 0, 0],     # Maroon
    [128, 128, 0],   # Olive
    [0, 128, 0],     # Dark Green
    [128, 0, 128],   # Purple
    [0, 128, 128],   # Teal
    [0, 0, 128],     # Navy
    [255, 165, 0],   # Orange
    [255, 192, 203], # Pink
    [165, 42, 42],   # Brown
    [255, 255, 255], # White
    [0, 0, 0],       # Black
    [127, 255, 212]  # Aquamarine
]

rows = 4
cols = 5

directory_path = "/home/mateo/uni/cuarto/VA/P2-Mateo_Amado/Material Plantones/"
directory_path_truth = "/home/mateo/uni/cuarto/VA/P2-Mateo_Amado/ground_truth/"

file_list = os.listdir(directory_path)
file_list_truth = os.listdir(directory_path_truth)

eval1 = []
eval2 = []

########################################################

for file in file_list:
    
    image_path = os.path.join(directory_path, file)
    image = io.imread(image_path)
    result_image = np.zeros((image.shape[0], image.shape[1]))
    #print(file)

    image_truth_path = os.path.join(directory_path_truth, file[:-4]+"_gt.png")
    image_truth = io.imread(image_truth_path)

    hsv_image = color.rgb2hsv(image)
    lower_green = np.array([0.1, 0.32, 0.22])  
    upper_green = np.array([0.5, 1.0, 1]) 
    green_mask = np.all((hsv_image >= lower_green) & (hsv_image <= upper_green), axis=-1)

    result_image = green_mask
    result_image = morphology.remove_small_holes(result_image.astype(bool), 150)    #performs better than closing
    result_image = morphology.remove_small_objects(result_image.astype(bool), 200)  #performs better than opening

#############################

    '''
    #aproximación por posición de las plantas
    green_partsbw = result_image.copy()
    green_parts = image.copy()
    green_parts[~green_partsbw] = 0
    label_image = measure.label(green_partsbw, connectivity=2)
    regions = measure.regionprops(label_image)
    object = np.zeros((image.shape[0], image.shape[1]))    

    cell_grid = [[None for _ in range(5)] for _ in range(4)]
    cell_truth = [[None for _ in range(5)] for _ in range(4)]
    plant_grid=np.zeros((rows, cols))
    size_grid=np.zeros((rows, cols))
    leafs_grid=np.zeros((rows, cols))

    # cant crop the image before since some leafes get cut
    crop=(40, image.shape[0]-40, 120, image.shape[1]-120)
    image_cropped = green_partsbw[crop[0]:crop[1], crop[2]:crop[3]]
    image_cropped_color = image[crop[0]:crop[1], crop[2]:crop[3]]
    image_truth_crop = image_truth[crop[0]:crop[1], crop[2]:crop[3]]
    #showImgs(image, image_cropped, "original")

    # height and width of each cell
    cell_height = image_cropped.shape[0] // rows
    cell_width = image_cropped.shape[1] // cols

    big_cell_height = image.shape[0] // rows
    big_cell_width = image.shape[1] // cols
    

    #este bucle recorre cada celda de la imagen y determina si hay planta y el tamaño 
    for i in range(rows):
        for j in range(cols):
            row_start = i * cell_height
            row_end = (i+1) * cell_height
            col_start = j * cell_width
            col_end = (j+1) * cell_width

            #center of the cell for more precision
            cell_center = image_cropped[row_start+100:row_end-100, col_start+100:col_end-100]
            cell_center_sum = np.sum(cell_center)
            #print(cell_center_sum)

            cell = image_cropped[row_start:row_end, col_start:col_end]
            cell_truth[i][j] = image_truth_crop[row_start:row_end, col_start:col_end]
            #result_image[row_start:row_end, col_start:col_end] = cell
            #showImgsgray(image_cropped, cell, "original")
            
            center_row = row_start + cell_height//2
            center_col = col_start + cell_width//2

            #si no hay suficiente verde en el centro de la celda, no hay planta
            if(cell_center_sum > 7000):
                plant_grid[i,j] = 1
                label_image = measure.label(cell, connectivity=2)
                regions = measure.regionprops(label_image)
                object = np.zeros((cell.shape[0], cell.shape[1]))    
                #print(regions)
                largest_region = max(regions, key=lambda region: region.area)
                largest_object = np.zeros_like(label_image)
                largest_object[label_image == largest_region.label] = 1
                #denoised_image = restoration.denoise_tv_chambolle(cell, weight=0.5)

                #comprobar si es necesario expandir la celda
                if(check_overflow(largest_object)>50):
                    plant_grid[i,j] = 2
                    big_cell = green_partsbw[max(row_start-50, 0):min(row_end+200, green_parts.shape[0]), max(col_start-150, 0):min(col_end+300, green_parts.shape[1])]
                    #result_image[max(row_start-50, 0):min(row_end+200, green_parts.shape[0]), max(col_start-150, 0):min(col_end+300, green_parts.shape[1])] = big_cell
                    big_cell_color = image[max(row_start-50, 0):min(row_end+200, green_parts.shape[0]), max(col_start-150, 0):min(col_end+300, green_parts.shape[1])]
                    big_label_image = measure.label(big_cell, connectivity=2)
                    big_regions = measure.regionprops(big_label_image)
                    big_object = np.zeros((big_cell.shape[0], big_cell.shape[1]))    

                    big_largest_region = max(big_regions, key=lambda region: region.area)
                    big_largest_object = np.zeros_like(big_label_image)
                    big_largest_object[big_label_image == big_largest_region.label] = 1
                    #crop=(40, image.shape[0]-40, 120, image.shape[1]-120)
                    showImgs(big_cell_color, big_largest_object, "og")
                    cell_grid[i][j]=big_cell_color
                    size_grid[i][j]=big_largest_region.area

                else:
                    showImgs(image_cropped_color[row_start:row_end, col_start:col_end], image_cropped_color, "og")
                    cell_grid[i][j]=image_cropped_color[row_start:row_end, col_start:col_end]
                    size_grid[i][j]=largest_region.area
    print(plant_grid)
    print(size_grid)
    '''

#############################
    #PCA_color_aug(image)
    #contrast_hsv(image)

    #result_image = filters.gaussian(result_image, sigma=0.5)
    #image[~green_mask]=0
    showImgs(image, result_image, "og")

    #segmentacion de las plantas mediante watershed
    #labels_plants=water(result_image_bool, 200)
    #diff = util.compare_images(color.rgb2gray(image_truth.astype(bool)), result_image.astype(bool), method='diff')
    #showImgs(image, diff, "og")

    #evaluación de la segmentación de plantas
    image_truthbw = color.rgb2gray(image_truth)
    result_image_bool = result_image > 0
    image_truth_bool = image_truthbw > 0
    true = image_truth_bool.flatten()
    pred = result_image_bool.flatten()
    showImgsgray(image_truth_bool, result_image_bool, "truth/result")

    ssim, precision, f1, recall, accuracy = eval(true, pred)
    eval1.append(["image:"+str(file_list.index(file)),ssim,precision,f1,recall,accuracy])

#############################

    #segmentacion de las hojas mediante watershed
    mindist=15
    #if np.count_nonzero(plant_grid == 2) > 5:
    #    mindist=25
    labels_leaves = water(result_image_bool, mindist)

    '''
    #Segmentacion de hojas por esqueleto
    #este bucle recorre cada celda y segmenta las hojas
    for i in range(rows):
        for j in range(cols):
            obj=cell_grid[i][j]
            if(plant_grid[i][j]==1):
                #showImgs(image, obj, "og")
                #print(size_grid[i][j])

                greened = isolate_green(obj)
                #isolate_contrast_green(obj)

                bw_obj = color.rgb2gray(greened)
                bin_obj = bw_obj > 0.02
                bin_obj = morphology.remove_small_objects(bin_obj, 150)
                bin_obj = morphology.remove_small_holes(bin_obj, 150)

                masked_image = obj.copy()
                masked_image[~bin_obj] = 0
                bw_obj = color.rgb2gray(masked_image)
                bw_obj = exposure.equalize_adapthist(bw_obj)

                #jugar con sigma y alpha para acertar más
                bw_obj = np.power(bw_obj, 2.5)
                edges = feature.canny(bw_obj, sigma=5)
                #contours = measure.find_contours(edges, level=0.8)
                paint_obj2 = obj.copy()
                paint_obj = np.zeros_like(obj)
                #skeleton = morphology.skeletonize(bw_obj)
                #thinned = morphology.thin(bw_obj)
                skel, distance = morphology.medial_axis(bin_obj, return_distance=True)
                dist_on_skel = distance * skel
                dist_on_skel[dist_on_skel < 12] = 0
                dist_on_skel= bin(dist_on_skel)
                dist_on_skel = morphology.remove_small_holes(dist_on_skel, 100)
                labels = measure.label(dist_on_skel, connectivity=2)
                regions = measure.regionprops(labels)
                #regions = remove_overlap(regions)

                showImgs(obj, dist_on_skel, "og")  
                color_i = 0
                leafs = []
                for region in regions:
                    xrt=25
                    minr, minc, maxr, maxc = region.bbox
                    leaf_region = obj[minr-xrt:maxr+xrt, minc-xrt:maxc+xrt]
                    bw_leaf_region = bw_obj[minr-xrt:maxr+xrt, minc-xrt:maxc+xrt]
                    bin_leaf_region = bin_obj[minr-xrt:maxr+xrt, minc-xrt:maxc+xrt]
                    leaf_edges = edges[minr-xrt:maxr+xrt, minc-xrt:maxc+xrt]
                    
                    #print(leaf_edges.shape)
                    labels = measure.label(leaf_edges, connectivity=2)
                    regions2 = measure.regionprops(labels) if labels.size > 0 else []
                    
                    filtered_regions = [region for region in regions2 if ((np.any(region.image)) & (region.area > 60)) ]
                    #showImgs(bw_leaf_region, leaf_edges, "leaf")  
                    for filtered_region in filtered_regions:
                        minr, minc, maxr, maxc = filtered_region.bbox
                        bin_leaf_region2 = bin_leaf_region[minr:maxr, minc:maxc]
                        #orientation = filtered_region.orientation
                        showImgs(leaf_edges, filtered_region.image, "leaf")  
                        padded_region2=pad_img(bin_leaf_region2, bin_leaf_region, filtered_region.bbox, 0)
                        padded_region=pad_img(padded_region2, obj, region.bbox, xrt)
                        paint_obj[padded_region] = colors[color_i%20]
                        paint_obj2[padded_region] = colors[color_i%20]
                        color_i+=1
                    leafs_grid[i][j]+=len(filtered_regions)
                cell_grid[i][j]=paint_obj
                showImgs(obj, paint_obj2, "og")
    
    #for i in range(rows):
    #    for j in range(cols):
    #        if(plant_grid[i][j]==1):
                #showImgs(cell_grid[i][j], cell_truth[i][j], "og")
    
    '''


    #evaluación
    edges1 = feature.canny(color.rgb2gray(color.label2rgb(labels_leaves)))
    edges2 = feature.canny(image_truthbw)
    true = edges2.flatten()
    pred = edges1.flatten()
    showImgsgray(edges2, edges1, "truth/result")
    ssim, precision, f1, recall, accuracy = eval(true, pred)
    eval2.append(["image:"+str(file_list.index(file)),ssim,precision,f1,recall,accuracy])

#############################

print(eval1)
print(eval2)

'''
#otro codigo que finalmente no fue utilizado

    #juntar regiones pequeñas del watershed
    small_regions = unique_labels[np.where(counts < 200)]
    labels[np.isin(labels, small_regions)] = -1
    for region in small_regions:
        if region == 0:
            continue
        neighbors = ndi.convolve((labels == region).astype(int), kernel)
        neighbors = labels * (neighbors > 0)
        neighbor_labels, neighbor_counts = np.unique(neighbors, return_counts=True)

        valid_neighbors = neighbor_labels != 0
        neighbor_labels = neighbor_labels[valid_neighbors]
        neighbor_counts = neighbor_counts[valid_neighbors]
        if len(neighbor_labels) == 0:
            continue
        most_frequent_neighbor = neighbor_labels[neighbor_counts.argmax()]
        labels[labels == region] = most_frequent_neighbor    


    BOUNDING BOXES
    label_image = measure.label(green_partsbw, connectivity=2)
    regions = measure.regionprops(label_image)
    bbImg = np.zeros((image.shape[0], image.shape[1]))    
    selected_regions = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        #rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        if (maxr - minr) * (maxc - minc) >= 6000:
            rr, cc = draw.rectangle(start=(minr, minc), end=(maxr, maxc), shape=(image.shape[0], image.shape[1]))
            bbImg[rr, cc] = 1  # Set pixel values to white
            selected_regions.append(region)
    print(len(selected_regions))
    showImgs(green_partsbw, bbImg, "original")


    pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=20, mask=bin_obj)
    leaf_obj, stem_obj= pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=bin_obj)
    filled_img = pcv.morphology.fill_segments(mask=bin_obj, objects=leaf_obj, label="default")
    showImgs(obj, filled_img, "og")  
    # https://plantcv.readthedocs.io/en/stable/tutorials/morphology_tutorial/
    # https://github.com/PRBonn/HAPT

    # Hough, https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html

    result = transform.hough_ellipse(edges, accuracy=20, threshold=250, min_size=10, max_size=200)
    result.sort(order='accumulator')

    best = list(result[-1])
    yc, xc, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]

    cy, cx = draw.ellipse_perimeter(yc, xc, a, b, orientation)
    obj[cy, cx] = (0, 0, 25
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                    sharex=True, sharey=True)

    ax1.set_title('Original picture')
    ax1.imshow(obj)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()


                
    ihc_hed = color.rgb2hed(obj) #? https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html

    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = color.hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = color.hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = color.hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    obj = np.float32(obj)
    obj = np.multiply(obj, 1)  
    obj = np.clip(obj, 0, 255)
    obj = np.uint8(obj)

    pixels = obj.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3) 
    kmeans.fit(pixels)

    labels = kmeans.labels_
    new_image = kmeans.cluster_centers_[labels]
    new_image = new_image.reshape(obj.shape)
    new_image = new_image.round(0).astype('uint8')
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

    #clustering sobre los puntos (kmeans, PCA)
    #producto escalar entre los vectores de los puntos

                
'''

########################################################