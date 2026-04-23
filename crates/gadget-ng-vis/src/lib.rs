//! `gadget-ng-vis` — Visualización de partículas N-body.
//!
//! ## Características
//!
//! - [`Renderer`]: renderiza partículas como puntos de 1 px sobre un canvas CPU RGBA.
//! - [`Projection`]: proyecciones ortográficas (XY, XZ, YZ) y perspectiva configurable.
//! - [`ColorMode`]: blanco uniforme, por velocidad o por densidad (Viridis).
//! - [`save_png`]: exporta cualquier frame a PNG vía la crate `png`.
//! - [`frame_path`]: genera el nombre de archivo `snap_{step:06}.png`.
//!
//! ## Ejemplo mínimo
//!
//! ```rust,no_run
//! use gadget_ng_vis::{Renderer, RendererConfig, Projection, ColorMode};
//! use gadget_ng_vis::export::frame_path;
//! use gadget_ng_core::Vec3;
//!
//! let cfg = RendererConfig {
//!     width: 800, height: 800, box_size: 100.0,
//!     projection: Projection::XY,
//!     color_mode: ColorMode::Velocity,
//! };
//! let mut renderer = Renderer::new(cfg);
//! let positions = vec![Vec3::new(10.0, 20.0, 0.0)];
//! let velocities = vec![Vec3::new(1.0, 0.0, 0.0)];
//! renderer.render_frame(&positions, &velocities);
//! renderer.save_frame(&frame_path(std::path::Path::new("frames"), 0)).unwrap();
//! ```
pub mod canvas;
pub mod color;
pub mod export;
pub mod ppm;
pub mod projection;
pub mod renderer;

pub use color::ColorMode;
pub use export::{frame_path, save_png};
pub use ppm::{render_density_ppm, render_ppm, render_ppm_projection, write_png, write_ppm};
pub use projection::Projection;
pub use renderer::{Renderer, RendererConfig};
