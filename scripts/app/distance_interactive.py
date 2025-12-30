import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from PyQt5.QtWidgets import QApplication, QFileDialog
from matplotlib.lines import Line2D 

from graph_neural_networks.reconstruction.reconstruction_method import PathReconstructionMethod
from graph_neural_networks.reconstruction.path_reconstruction.euclidean_path_reconstruction import EuclideanPathReconstructionMethod
from graph_neural_networks.reconstruction.path_reconstruction.min_energy_path_reconstruction import ClassicMinEnergyPathReconstructionMethod, SquaredMinEnergyPathReconstructionMethod
from graph_neural_networks.reconstruction.path_reconstruction.dahu_distance_path_reconstruction import DahuDistancePathReconstructionMethod

def enable_scroll_zoom(fig, ax):
    """Enable zooming in/out with the mouse wheel."""
    def on_scroll(event):
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return  # Ignore if mouse outside axes

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        scale_factor = 1.2 if event.button == 'up' else 0.8

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])
        ax.set_ylim([ydata - new_height * rely, ydata + new_height * (1 - rely)])
        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

class TwoPointsApp:
    def __init__(self, path=None):
        self.path_methods: dict[str, PathReconstructionMethod] = {
            "euclidean": EuclideanPathReconstructionMethod(),
            #"classic_min_energy": ClassicMinEnergyPathReconstructionMethod(),
            #"squared_min_energy": SquaredMinEnergyPathReconstructionMethod(),
            "dahu_distance": DahuDistancePathReconstructionMethod(),
        }

        if path is None:
            image_path = self.choose_file()
        else:
            image_path = path
        if image_path is None:
            sys.exit()

        # Gestion des fichiers .pt ou images classiques
        if image_path.endswith(".pt"):
            self.image = self.load_prob_map(image_path)
        else:
            self.image = self.load_image(image_path)

        self.points = []

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.ax.set_title("Clique sur deux points")

        enable_scroll_zoom(self.fig, self.ax)

        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event",
            self.onclick
        )

        plt.show()

    def choose_file(self):
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Choisis une image ou un fichier .pt",
            "",
            "Images et fichiers PyTorch (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pt)"
        )

        if file_path == "":
            return None

        return file_path

    def load_image(self, path):
        image = mpimg.imread(path)
        if image.ndim == 3:
            image = image[:, :, 0] # Convert to grayscale if RGB
        return image

    def load_prob_map(self, path):
        prob_map = self.load_torch_tensor(path)
        if prob_map.ndim == 3 and prob_map.shape[2] == 2:
            prob_map = prob_map[:, :, 1]
        return prob_map

    def load_torch_tensor(self, path):
        tensor = torch.load(path)
        # Si le tensor a plusieurs canaux (C,H,W) ou (H,W,C), on le met en HxW ou HxWx3
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] <= 4:  # C,H,W → H,W,C
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.ndim == 2:  # H,W
                pass
            else:
                raise ValueError("Format du tensor inattendu pour l'affichage")
            # Normalisation si float
            if arr.dtype in [np.float32, np.float64]:
                arr = arr - arr.min()
                if arr.max() != 0:
                    arr = arr / arr.max()
            return arr
        else:
            raise ValueError("Le fichier .pt ne contient pas un tensor PyTorch valide")

    def onclick(self, event):
        if event.inaxes != self.ax:
            return

        self.points.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, "ro")
        self.fig.canvas.draw()

        if len(self.points) == 2:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.process_points()

    def process_points(self):
        p1 = np.array(self.points[0])
        p2 = np.array(self.points[1])

        final_image = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        if self.image.ndim == 2:
            final_image[:, :, 0] = (self.image * 255).astype(np.uint8)
            final_image[:, :, 1] = (self.image * 255).astype(np.uint8)
            final_image[:, :, 2] = (self.image * 255).astype(np.uint8)
        else:
            final_image = (self.image * 255).astype(np.uint8)

        colormap = plt.get_cmap('hsv')
        n_methods = len(self.path_methods)
        colors = [colormap(i / n_methods) for i in range(n_methods)]

        legend_lines = []
        legend_labels = []

        for i, (method_name, method) in enumerate(self.path_methods.items()):
            color = (np.array(colors[i]) * 255)[:3].astype(np.uint8)
            start = (int(round(p1[1])), int(round(p1[0])))
            goal = (int(round(p2[1])), int(round(p2[0])))
            path = method.reconstruct_one(torch.tensor(self.image), start, goal)
            path = [(pt[1], pt[0]) for pt in path]  # Inverser les coordonnées pour (x,y)
            for (x, y) in path:
                x = int(round(x))
                y = int(round(y))
                if 0 <= x < final_image.shape[1] and 0 <= y < final_image.shape[0]:
                    final_image[y, x, :] = color

            # Prepare legend entry
            legend_lines.append(Line2D([0], [0], color=np.array(colors[i]), lw=2))
            legend_labels.append(method_name)

        self.fig_result, self.ax_result = plt.subplots()
        self.ax_result.imshow(final_image)

        enable_scroll_zoom(self.fig_result, self.ax_result)

        # Add legend for methods
        self.ax_result.legend(legend_lines, legend_labels, loc='upper right')

        self.ax_result.set_title("Résultat")

        self.fig_result.canvas.mpl_connect(
            "close_event",
            self.on_result_close
        )

        plt.show()


    def on_result_close(self, event):
        self.points = []

        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title("Clique sur deux points")

        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event",
            self.onclick
        )

        self.fig.canvas.draw()


if __name__ == "__main__":
    TwoPointsApp()
    #TwoPointsApp(path="/home/morand/afs/EVAPORE/data/FIVES/probability_maps/FIVES_001.pt")
    #TwoPointsApp(path="/home/morand/afs/EVAPORE/scripts/app/curved_up.png")
    #TwoPointsApp(path="/home/morand/afs/EVAPORE/scripts/app/up.png")
    #TwoPointsApp(path="/home/morand/afs/EVAPORE/scripts/app/double.png")