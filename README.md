
<a id="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## About The Project
Look! Jon Jones only has a 32% chance of beating Stipe Miocic for the upcoming UFC 309!
<p align="center">
  <img width="928" alt="Screenshot 2024-08-18 at 6 49 13 PM" src="https://github.com/user-attachments/assets/bd24c51d-0bef-493d-b0c9-16ada14c378a"
">
</p>

<p>
A client built on the UFC predictor microservice available in this repository: https://github.com/Ch3mson/ufc-predictor. This app allows anyone on the web to access the ML model that I created to predict on upcoming UFC fights, for both men and women. For best results, choose fighters that are in the same weight division and gender, since the model is trained on fight stats, and doesn't take into account of weight/gender. On startup, it may take a few seconds since Google Cloud Run needs to do a cold start when it is inactive.
</p>

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Ch3mson/ufc-client.git
   ```
2. Change directory to project and ensure npm and node.js are installed
   ```sh
   npm install
   ```
3. Run the server and do as you wish
   ```sh
   npm run dev
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Inspiration](https://www.youtube.com/watch?v=0irmDBWLrco)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
