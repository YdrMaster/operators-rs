mod gguf;

use digit_layout::DigitLayout;
use gguf::{Content, GGufTensor};
use ggus::{GGufMetaDataValueType, GGufMetaMap, GGufMetaMapExt};
use memmap2::Mmap;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::{collections::HashMap, env::var_os, fs::File};

/// 测例数据。
pub struct TestCase<'a> {
    /// 测例在文件中的序号。
    pub index: usize,
    /// 测例元信息，即传递给算子非张量参数。
    pub attributes: HashMap<String, MetaValue<'a>>,
    /// 测例张量，包括算子的输入和正确答案。
    pub tensors: HashMap<String, Tensor<'a>>,
}

/// 元信息键值对。
pub struct MetaValue<'a> {
    pub ty: GGufMetaDataValueType,
    pub value: &'a [u8],
}

/// 测例张量。
pub struct Tensor<'a> {
    pub ty: DigitLayout,
    pub layout: ArrayLayout<4>,
    pub data: &'a [u8],
}

impl GGufMetaMap for TestCase<'_> {
    fn get(&self, key: &str) -> Option<(GGufMetaDataValueType, &[u8])> {
        self.attributes.get(key).map(|v| (v.ty, &*v.value))
    }
}

impl<'a> Content<'a> {
    pub fn into_cases(self) -> HashMap<String, Vec<TestCase<'a>>> {
        assert_eq!(self.general_architecture().unwrap(), "infiniop-test");
        let mut ans = HashMap::new();

        let ntest = self.get_usize("test_count").unwrap();
        let Self {
            mut meta_kvs,
            mut tensors,
        } = self;
        for i in 0..ntest {
            let prefix = format!("test.{i}.");

            let mut meta_kvs = meta_kvs
                .split_by_prefix(&prefix)
                .into_iter()
                .map(|(k, v)| (k[prefix.len()..].to_string(), v))
                .collect::<HashMap<_, _>>();
            let tensors = tensors
                .split_by_prefix(&prefix)
                .into_iter()
                .map(|(k, v)| {
                    let GGufTensor {
                        ty,
                        mut shape,
                        data,
                    } = v;
                    shape.reverse();

                    let k = k[prefix.len()..].to_string();
                    let element_size = ty.nbytes();
                    let layout = if let Some(strides) = meta_kvs.remove(&format!("{k}.strides")) {
                        let mut strides = strides.to_vec_isize().unwrap();
                        for x in &mut strides {
                            *x *= element_size as isize
                        }
                        strides.reverse();

                        ArrayLayout::<4>::new(&shape, &strides, 0)
                    } else {
                        ArrayLayout::<4>::new_contiguous(&shape, BigEndian, element_size)
                    };
                    (k, Tensor { ty, layout, data })
                })
                .collect();

            let case = TestCase {
                index: i,
                attributes: meta_kvs,
                tensors,
            };

            let op_name = case.get_str("op_name").unwrap().to_string();
            ans.entry(op_name).or_insert_with(Vec::new).push(case);
        }

        ans
    }
}

#[test]
fn test() {
    let Some(name) = var_os("TEST_CASES") else {
        eprintln!("TEST_CASES not set");
        return;
    };
    let Ok(file) = File::open(&name) else {
        eprintln!("Failed to open {}", name.to_string_lossy());
        return;
    };
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let cases = Content::new(&[&*mmap]).unwrap().into_cases();

    for (op_name, cases) in cases {
        for case in cases {
            println!("Test case {}: {op_name}", case.index);
            for (k, v) in &case.attributes {
                println!("  {k}: {:?}", v.ty)
            }
            for (k, v) in &case.tensors {
                println!(
                    "  {k}: {} {:?} / {:?} {}",
                    v.ty,
                    v.layout.shape(),
                    v.layout.strides(),
                    v.data.len()
                )
            }
        }
    }
}
