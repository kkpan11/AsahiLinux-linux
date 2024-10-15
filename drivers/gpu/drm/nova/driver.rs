// SPDX-License-Identifier: GPL-2.0

use kernel::{
    bindings, c_str, drm,
    drm::{drv, ioctl},
    pci,
    prelude::*,
    sync::Arc,
};

use crate::{file::File, gpu::Gpu};

pub(crate) struct NovaDriver {
    _reg: drm::drv::Registration<Self>,
}

/// Convienence type alias for the DRM device type for this driver
pub(crate) type NovaDevice = drm::device::Device<NovaDriver>;

#[allow(dead_code)]
#[pin_data]
pub(crate) struct NovaData {
    #[pin]
    pub(crate) gpu: Gpu,
    pub(crate) pdev: pci::Device,
}

const BAR0_SIZE: usize = 8;
pub(crate) type Bar0 = pci::Bar<BAR0_SIZE>;

const INFO: drm::drv::DriverInfo = drm::drv::DriverInfo {
    major: 0,
    minor: 0,
    patchlevel: 0,
    name: c_str!("nova"),
    desc: c_str!("Nvidia Graphics"),
    date: c_str!("20240227"),
};

kernel::pci_device_table!(
    PCI_TABLE,
    MODULE_PCI_TABLE,
    <NovaDriver as pci::Driver>::IdInfo,
    [(
        pci::DeviceId::from_id(bindings::PCI_VENDOR_ID_NVIDIA, bindings::PCI_ANY_ID as _),
        ()
    )]
);

impl pci::Driver for NovaDriver {
    type IdInfo = ();
    const ID_TABLE: pci::IdTable<Self::IdInfo> = &PCI_TABLE;

    fn probe(pdev: &mut pci::Device, _info: &Self::IdInfo) -> Result<Pin<KBox<Self>>> {
        dev_dbg!(pdev.as_ref(), "Probe Nova GPU driver.\n");

        pdev.enable_device_mem()?;
        pdev.set_master();

        let bar = pdev.iomap_region_sized::<BAR0_SIZE>(0, c_str!("nova"))?;

        let p = pdev.clone();
        let data = Arc::pin_init(
            try_pin_init!(NovaData {
                gpu <- Gpu::new(&p, bar)?,
                pdev: p,
            }),
            GFP_KERNEL,
        )?;

        let drm = drm::device::Device::<Self>::new(pdev.as_ref())?;
        let reg = drm::drv::Registration::new(drm.clone(), data, 0)?;

        Ok(KBox::new(Self { _reg: reg }, GFP_KERNEL)?.into())
    }
}

#[vtable]
impl drm::drv::Driver for NovaDriver {
    type Data = Arc<NovaData>;
    type File = File;
    type Object = crate::gem::Object;

    const INFO: drm::drv::DriverInfo = INFO;
    const FEATURES: u32 = drv::FEAT_GEM;

    kernel::declare_drm_ioctls! {
        (NOVA_GETPARAM, drm_nova_getparam, ioctl::RENDER_ALLOW, File::get_param),
        (NOVA_GEM_CREATE, drm_nova_gem_create, ioctl::AUTH | ioctl::RENDER_ALLOW, File::gem_create),
        (NOVA_GEM_INFO, drm_nova_gem_info, ioctl::AUTH | ioctl::RENDER_ALLOW, File::gem_info),
    }
}
