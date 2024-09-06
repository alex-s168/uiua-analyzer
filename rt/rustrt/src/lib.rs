extern crate core;

use core::slice;
use std::ffi::{c_char, c_double, c_void, CStr};
use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut};
use ecow::EcoVec;
use uiua::{Array, Assembly, Boxed, Shape, Uiua, Value};

#[repr(C)]
#[derive(Clone)]
pub struct ArrU<T> {
    rank: usize,
    alloc: *mut c_void,
    aligned: *mut T,
    elems_off: usize,
    sizes: *mut usize,
    strides: *mut usize,
}

pub trait ArrUAlloc<T> {
    fn alloc(shape: &[usize]) -> ArrU<T>;
}

impl ArrUAlloc<i64> for ArrU<i64> {
    fn alloc(shape: &[usize]) -> ArrU<i64> {
        unsafe {
            allocUArr_int64_t(shape.iter().product(),
                              shape.len(),
                              shape.as_ptr())
        }
    }
}

impl ArrUAlloc<usize> for ArrU<usize> {
    fn alloc(shape: &[usize]) -> ArrU<usize> {
        unsafe {
            allocUArr_size_t(shape.iter().product(),
                              shape.len(),
                              shape.as_ptr())
        }
    }
}

impl ArrUAlloc<UACVal> for ArrU<UACVal> {
    fn alloc(shape: &[usize]) -> ArrU<UACVal> {
        unsafe {
            allocUArr_uac_Dyn(shape.iter().product(),
                             shape.len(),
                             shape.as_ptr())
        }
    }
}

impl ArrUAlloc<c_double> for ArrU<c_double> {
    fn alloc(shape: &[usize]) -> ArrU<c_double> {
        unsafe {
            allocUArr_double(shape.iter().product(),
                             shape.len(),
                             shape.as_ptr())
        }
    }
}

impl ArrUAlloc<u8> for ArrU<u8> {
    fn alloc(shape: &[usize]) -> ArrU<u8> {
        unsafe {
            allocUArr_uint8_t(shape.iter().product(),
                              shape.len(),
                              shape.as_ptr())
        }
    }
}

fn free_arr<T>(arr: ArrU<T>) {
    unsafe { arrDeallocImpl(arr.alloc) };
}

fn copy_slice_into_arr<T, R>(slice: &[T], into: &ArrU<R>, map: impl Fn(&T) -> R) {
    let shape = unsafe { slice_from_raw_parts(into.sizes, into.rank)
        .as_ref().unwrap_unchecked() };

    let shape_product = shape.iter().product();

    let dest = unsafe { slice_from_raw_parts_mut(into.aligned.offset(into.elems_off as isize), shape_product)
            .as_mut().unwrap_unchecked() };

    let strides = unsafe { slice_from_raw_parts(into.strides, into.rank)
        .as_ref().unwrap_unchecked() };

    let stride = strides.last()
        .unwrap();

    let mut did = 0;
    for elem in slice {
        dest[did] = map(elem);
        did += stride;
    }
}

fn copy_uiua_into_arr<T: Sync + Send, R>(uiua: Array<T>, into: &ArrU<R>, map: impl Fn(&T) -> R) {
    let shape_product = uiua.shape().iter().product();

    let dest = unsafe { slice_from_raw_parts_mut(into.aligned.offset(into.elems_off as isize), shape_product)
        .as_mut().unwrap_unchecked() };

    let strides = unsafe { slice_from_raw_parts(into.strides, into.rank)
        .as_ref().unwrap_unchecked() };

    let stride = strides.last()
        .unwrap();

    let mut did = 0;
    for row in uiua.row_slices() {
        for elem in row {
            dest[did] = map(elem);
            did += stride;
        }
    }
}

fn copy_uiua_to_arr<T: Sync + Send, R>(arr: Array<T>, map: impl Fn(&T) -> R) -> ArrU<R>
    where ArrU<R>: ArrUAlloc<R>
{
    let res = ArrU::alloc(arr.shape().dims());
    copy_uiua_into_arr(arr, &res, map);
    res
}

fn copy_slice_to_arr<T, R>(slice: &[T], map: impl Fn(&T) -> R) -> ArrU<R>
    where ArrU<R>: ArrUAlloc<R>
{
    let res = ArrU::alloc(&[slice.len()]);
    copy_slice_into_arr(slice, &res, map);
    res
}

fn copy_arr_to_uiua<T, R: Clone>(arr: ArrU<T>, into: impl Fn(&T) -> R) -> Array<R> {
    let shape_slice = unsafe { slice_from_raw_parts(arr.sizes, arr.rank)
        .as_ref().unwrap_unchecked() };

    let strides_slice = unsafe { slice_from_raw_parts(arr.strides, arr.rank)
        .as_ref().unwrap_unchecked() };

    let shape_product = shape_slice.iter().product();

    let mut data = EcoVec::with_capacity(shape_product);

    let input = unsafe {
        // can't specify shape_product as len because strides
        slice_from_raw_parts(arr.aligned.add(arr.elems_off), 69)
            .as_ref()
            .unwrap_unchecked()
    };

    let stride = unsafe { strides_slice.last().unwrap_unchecked() };

    for i in 0..shape_product {
        // need get unchecked because we set len to 69
        let x = unsafe { input.get_unchecked(i * stride) };
        data.push(into(x))
    }

    Array::new(Shape::from(shape_slice), data)
}

type UACValKind = u8;

const UAC_NOTYPE: UACValKind = 0;
const UAC_BYTE: UACValKind = 1;
const UAC_INT: UACValKind  = 2;
const UAC_FLT: UACValKind  = 3;
const UAC_SIZE: UACValKind = 4;
const UAC_DYN: UACValKind = 5;
const UAC_ARR_BYTE: UACValKind = 6;
const UAC_ARR_INT: UACValKind  = 7;
const UAC_ARR_FLT: UACValKind  = 8;
const UAC_ARR_SIZE: UACValKind = 9;
const UAC_ARR_DYN: UACValKind = 10;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct UACVal {
    kind: UACValKind,
    opaque: *const c_void,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct LightCArr<T> {
    len: usize,
    ptr: *mut T,
}

impl<T> LightCArr<T> {
    fn into_iter(self) -> slice::IterMut<'static, T> {
        unsafe {
            slice_from_raw_parts_mut(self.ptr, self.len)
                .as_mut()
                .unwrap_unchecked()
                .iter_mut()
        }
    }
}

extern "C" {
    fn arrDeallocImpl(v: *mut c_void);

    fn uac_Dyn_as_uint8_t(v: UACVal) -> u8;
    fn uac_Dyn_as_int64_t(v: UACVal) -> i64;
    fn uac_Dyn_as_double(v: UACVal) -> c_double;
    fn uac_Dyn_as_size_t(v: UACVal) -> usize;
    fn uac_Dyn_as_uac_Dyn(v: UACVal) -> UACVal;

    fn uac_Dyn_as_Arru_uint8_t(v: UACVal) -> ArrU<u8>;
    fn uac_Dyn_as_Arru_int64_t(v: UACVal) -> ArrU<i64>;
    fn uac_Dyn_as_Arru_double(v: UACVal) -> ArrU<c_double>;
    fn uac_Dyn_as_Arru_size_t(v: UACVal) -> ArrU<usize>;
    fn uac_Dyn_as_Arru_uac_Dyn(v: UACVal) -> ArrU<UACVal>;

    fn uac_Dyn_from_uint8_t(v: u8) -> UACVal;
    fn uac_Dyn_from_int64_t(v: i64) -> UACVal;
    fn uac_Dyn_from_double(v: c_double) -> UACVal;
    fn uac_Dyn_from_size_t(v: usize) -> UACVal;
    fn uac_Dyn_from_uac_Dyn(v: UACVal) -> UACVal;

    fn uac_Dyn_from_Arru_uint8_t(v: ArrU<u8>) -> UACVal;
    fn uac_Dyn_from_Arru_int64_t(v: ArrU<i64>) -> UACVal;
    fn uac_Dyn_from_Arru_double(v: ArrU<c_double>) -> UACVal;
    fn uac_Dyn_from_Arru_size_t(v: ArrU<usize>) -> UACVal;
    fn uac_Dyn_from_Arru_uac_Dyn(v: ArrU<UACVal>) -> UACVal;

    fn allocUArr_uint8_t(num_elem: usize, rank: usize, sizes_in: *const usize) -> ArrU<u8>;
    fn allocUArr_int64_t(num_elem: usize, rank: usize, sizes_in: *const usize) -> ArrU<i64>;
    fn allocUArr_double(num_elem: usize, rank: usize, sizes_in: *const usize) -> ArrU<c_double>;
    fn allocUArr_size_t(num_elem: usize, rank: usize, sizes_in: *const usize) -> ArrU<usize>;
    fn allocUArr_uac_Dyn(num_elem: usize, rank: usize, sizes_in: *const usize) -> ArrU<UACVal>;

    fn uac_Dyn_drop(v: UACVal);
}

fn from_uiua(val: Value) -> Option<UACVal> {
    match val {
        Value::Byte(arr) => unsafe {
            if arr.rank() == 0 || (arr.rank() == 1 && arr.shape()[0] == 1) {
                Some(uac_Dyn_from_uint8_t(arr.row_slice(0)[0]))
            } else {
                Some(uac_Dyn_from_Arru_uint8_t(copy_uiua_to_arr(arr, |x| *x)))
            }
        }

        Value::Num(arr) => unsafe {
            if arr.rank() == 0 || (arr.rank() == 1 && arr.shape()[0] == 1) {
                Some(uac_Dyn_from_double(arr.row_slice(0)[0]))
            } else {
                Some(uac_Dyn_from_Arru_double(copy_uiua_to_arr(arr, |x| *x)))
            }
        }

        Value::Char(arr) => unsafe {
            if arr.rank() == 0 || (arr.rank() == 1 && arr.shape()[0] == 1) {
                Some(uac_Dyn_from_int64_t(arr.row_slice(0)[0] as u32 as i64))
            } else if arr.rank() == 1 {
                Some(uac_Dyn_from_Arru_uint8_t(copy_slice_to_arr(
                    arr.row_slice(0).iter().collect::<String>().as_bytes(),
                    |x| *x)))
            } else {
                // chars are unicode and if we convert to utf8 then rows might have different len
                // that's why we convert

                Some(uac_Dyn_from_Arru_int64_t(copy_uiua_to_arr(arr, |x| *x as i32 as i64)))
            }
        }

        Value::Box(arr) => unsafe {
            if arr.rank() == 0 || (arr.rank() == 1 && arr.shape()[0] == 1) {
                Some(uac_Dyn_from_uac_Dyn(from_uiua(
                    arr.row_slice(0)[0].0.clone())?))
            } else {
                Some(uac_Dyn_from_Arru_uac_Dyn(
                    copy_uiua_to_arr(arr,|x| from_uiua(x.clone().0).unwrap())))
            }
        }

        _ => None
    }
}

fn to_uiua(val: UACVal) -> Option<Value> {
    match val.kind {
        UAC_BYTE => Some(Value::Byte(
            Array::scalar(
                unsafe { uac_Dyn_as_uint8_t(val) }))),

        UAC_INT => Some(Value::Num(
            Array::scalar(
                unsafe { uac_Dyn_as_int64_t(val) } as f64))),

        UAC_FLT => Some(Value::Num(
            Array::scalar(
                unsafe { uac_Dyn_as_double(val) } as f64))),

        UAC_SIZE => Some(Value::Num(
            Array::scalar(
                unsafe { uac_Dyn_as_size_t(val) } as f64))),

        UAC_DYN => Some(Value::Box(
            Array::scalar(
                Boxed(to_uiua(unsafe { uac_Dyn_as_uac_Dyn(val) })?))
        )),

        UAC_ARR_BYTE => Some(Value::Byte(
            copy_arr_to_uiua(unsafe { uac_Dyn_as_Arru_uint8_t(val) }, |x| *x))),

        UAC_ARR_INT => Some(Value::Num(
            copy_arr_to_uiua(unsafe { uac_Dyn_as_Arru_int64_t(val) }, |x| *x as f64))),

        UAC_ARR_FLT => Some(Value::Num(
            copy_arr_to_uiua(unsafe { uac_Dyn_as_Arru_double(val) }, |x| *x))),

        UAC_ARR_SIZE => Some(Value::Num(
            copy_arr_to_uiua(unsafe { uac_Dyn_as_Arru_size_t(val) }, |x| *x as f64))),

        UAC_ARR_DYN => Some(Value::Box(
            copy_arr_to_uiua(unsafe { uac_Dyn_as_Arru_uac_Dyn(val) },
                             |x| Boxed(to_uiua(*x).unwrap()))
        )),

        _ => None
    }
}

#[no_mangle]
pub extern "C" fn uac_interpretImpl(uasm_path: *const c_char,
                                    instr_first: usize,
                                    instr_last: usize,
                                    args: LightCArr<UACVal>,
                                    rets: LightCArr<UACVal>) {

    let uasm_path_rs = unsafe { CStr::from_ptr(uasm_path)
        .to_str().unwrap_unchecked() };

    let mut asm = Assembly::from_uasm(uasm_path_rs)
        .unwrap();

    let mut uiua = Uiua::with_native_sys();

    for x in args.into_iter() {
        uiua.push(to_uiua(x.clone()).unwrap());
        unsafe { uac_Dyn_drop(x.clone()) };
    }

    // TODO: run from instr_first to instr_last using magic

    if uiua.stack().len() != rets.len {
        panic!("too many / few values on stack")
    }

    for x in rets.into_iter() {
        *x = from_uiua(uiua.pop(0).unwrap()).unwrap()
    }
}