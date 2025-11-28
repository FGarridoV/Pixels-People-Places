import numpy as np


def create_section_vectors(image_assignments: dict[str, list[str]], project_name: str) -> dict[str, np.ndarray]:
    section_vectors = {}

    ten_percent = int(np.ceil([len(image_assignments) / 10])[0])
    count = 0
    for section, images in image_assignments.items():
        all_arrays = []

        for image in images:
            array = np.load(f"Intermediate Data/{project_name}/image_features/{image}.npy")

            all_arrays.append(array)

        mean_array = np.mean(np.array(all_arrays), axis=0)

        section_vectors[section] = mean_array
        count += 1

        if count % ten_percent == 0:
            print(f"{count} out of {len(image_assignments)} section feature vectors created")
    
    print(f"All {len(image_assignments)} section feature vectors created")

    return section_vectors




def create_section_vectors_front_and_back_seperated(image_assignments: dict[str, list[str]],
                                                    project_name: str) -> dict[str, np.ndarray]:
    section_vectors = {}

    ten_percent = int(np.ceil([len(image_assignments) / 10])[0])
    count = 0
    for section, images in image_assignments.items():
        front_arrays = []
        back_arrays = []

        for image in images:
            array = np.load(f"Intermediate Data/{project_name}/image_features/{image}.npy")

            if image.endswith('_f.png'):
                front_arrays.append(array)
            else:  # image.endswith('_b.png'):
                back_arrays.append(array)

        front_array = np.mean(np.array(front_arrays), axis=0)
        back_array = np.mean(np.array(back_arrays), axis=0)

        section_vectors[section] = np.concatenate((front_array, back_array), axis=0)
        count += 1

        if count % ten_percent == 0:
            print(f"{count} out of {len(image_assignments)} section feature vectors created")
    
    print(f"All {len(image_assignments)} section feature vectors created")

    return section_vectors