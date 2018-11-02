import numpy as np

class GIF_Helper(object):
    def __init__(self, image, predictions):
        # uses image and predictions to create animations
        # predictions is a BoxList() holding only the items to possibly animate
        # coco_demo is of the class COCODemo

        # should be a bgr image
        self.image = image
        self.predictions = predictions

        self.height, self.width = self.image.shape[:2]

        # set the masks
        self.masks = self.predictions.get_field("mask")

        self.num_predictions = len(predictions)

    def get_center_pivot_from_bounding_box(self, box):
        # return the center of the bounding box
        x1, y1, x2, y2 = box
        return (int((x1+x2)/2.0), int((y1+y2)/2.0))
    
    def get_points_from_foreground_image(self, foreground):
        # returns a dictionary of x,y points and pixel values
        point_dict = {}
        for y in range(self.height):
            for x in range(self.width):
                if np.sum(foreground[y, x, :]) != 0:
                    point_dict[(x,y)] = foreground[y, x, :]
        return point_dict

    def draw_points_on_image(self, image, points):
        for key, value in points.items():
            y = key[1]
            x = key[0]
            if 0 <= x < self.width and 0 <= y < self.height:
                image[y, x, :] = value

    def get_box_at_index(self, index):
        return self.predictions.bbox[index]
    
    def get_mask_at_index(self, index):
        return self.masks[index][0, :, :].numpy()