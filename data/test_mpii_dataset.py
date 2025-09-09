from mpii_dataset import MPII_Dataset
import matplotlib.pyplot as plt

face_dataset = MPII_Dataset(images_path='cropped_persons/',
                            annotations_path='annotations/cropped_persons_annotations.json',
                                    )

fig = plt.figure()

def show_joints(image, joints):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(joints[:, 0], joints[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

for i, sample in enumerate(face_dataset):
    print(i, sample['image'].shape, sample['joints'].shape)

    ax = plt.subplot(1, 4, i + 1)
    #plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_joints(**sample)

    if i == 3:
        plt.show()
        break