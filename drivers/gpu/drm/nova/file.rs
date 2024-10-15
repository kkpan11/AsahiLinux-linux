// SPDX-License-Identifier: GPL-2.0

use crate::driver::{NovaDevice, NovaDriver};
use crate::gem;
use kernel::{
    alloc::flags::*,
    drm::{self, device::Device as DrmDevice, gem::BaseObject},
    prelude::*,
    types::ForeignOwnable,
    uapi,
};

pub(crate) struct File;

/// Convenience type alias for our DRM `File` type
pub(crate) type DrmFile = drm::file::File<File>;

impl drm::file::DriverFile for File {
    kernel::define_driver_file_types!(NovaDriver);

    fn open(
        dev: &DrmDevice<Self::Driver>,
        _data: <Self as drm::file::DriverFile>::BorrowedData<'_>,
    ) -> Result<Pin<KBox<Self>>> {
        dev_dbg!(dev.as_ref(), "drm::device::Device::open\n");

        Ok(KBox::new(Self, GFP_KERNEL)?.into())
    }
}

impl File {
    /// IOCTL: get_param: Query GPU / driver metadata.
    pub(crate) fn get_param(
        _dev: &NovaDevice,
        data: <Self as drm::file::DriverFile>::BorrowedData<'_>,
        getparam: &mut uapi::drm_nova_getparam,
        _file: &DrmFile,
    ) -> Result<u32> {
        let pdev = &data.pdev;

        getparam.value = match getparam.param as u32 {
            uapi::NOVA_GETPARAM_VRAM_BAR_SIZE => pdev.resource_len(1)?,
            _ => return Err(EINVAL),
        };

        Ok(0)
    }

    /// IOCTL: gem_create: Create a new DRM GEM object.
    pub(crate) fn gem_create(
        dev: &NovaDevice,
        _data: <Self as drm::file::DriverFile>::BorrowedData<'_>,
        req: &mut uapi::drm_nova_gem_create,
        file: &DrmFile,
    ) -> Result<u32> {
        let obj = gem::object_new(dev, req.size.try_into()?)?;

        let handle = obj.create_handle(file)?;
        req.handle = handle;

        Ok(0)
    }

    /// IOCTL: gem_info: Query GEM metadata.
    pub(crate) fn gem_info(
        _dev: &NovaDevice,
        _data: <Self as drm::file::DriverFile>::BorrowedData<'_>,
        req: &mut uapi::drm_nova_gem_info,
        file: &DrmFile,
    ) -> Result<u32> {
        let bo = gem::lookup_handle(file, req.handle)?;

        req.size = bo.size().try_into()?;

        Ok(0)
    }
}
