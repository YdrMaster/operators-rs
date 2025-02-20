use super::MetaValue;
use digit_layout::DigitLayout;
use ggus::{
    GGuf, GGufError, GGufMetaDataValueType, GGufMetaError, GGufMetaMap, GGufMetaValueArray,
    GGufReader, GENERAL_ALIGNMENT,
};
use patricia_tree::StringPatriciaMap;

/// GGuf 文件内容，元信息和张量。
pub(super) struct Content<'a> {
    pub meta_kvs: StringPatriciaMap<MetaValue<'a>>,
    pub tensors: StringPatriciaMap<GGufTensor<'a>>,
}

impl GGufMetaMap for Content<'_> {
    fn get(&self, key: &str) -> Option<(GGufMetaDataValueType, &[u8])> {
        self.meta_kvs.get(key).map(|v| (v.ty, &*v.value))
    }
}

#[derive(Clone, Debug)]
pub(super) struct GGufTensor<'a> {
    pub ty: DigitLayout,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl<'a> Content<'a> {
    /// 从分片的 GGuf 文件解析内容。
    pub fn new(files: &[&'a [u8]]) -> Result<Self, GGufError> {
        std::thread::scope(|s| {
            let mut ans = Self {
                meta_kvs: Default::default(),
                tensors: Default::default(),
            };
            // 在多个线程中并行解析多个文件，并逐个合并到单独的结构体中
            for thread in files
                .into_iter()
                .map(|data| s.spawn(|| GGuf::new(data)))
                .collect::<Vec<_>>()
                .into_iter()
            {
                thread
                    .join()
                    .unwrap()
                    .and_then(|gguf| ans.merge_file(gguf))?;
            }

            Ok(ans)
        })
    }

    fn merge_file(&mut self, others: GGuf<'a>) -> Result<(), GGufError> {
        // 合并元信息
        for (k, kv) in others.meta_kvs {
            if k == GENERAL_ALIGNMENT || k.starts_with("split.") {
                continue;
            }
            let value = MetaValue {
                ty: kv.ty(),
                value: kv.value_bytes(),
            };
            if self.meta_kvs.insert(k.to_string(), value).is_some() {
                return Err(GGufError::DuplicateMetaKey(k.into()));
            }
        }
        // 合并张量，并将形状转换到 usize 类型
        for (name, tensor) in others.tensors {
            let tensor = tensor.to_info();
            let tensor = GGufTensor {
                ty: tensor.ty().to_digit_layout(),
                shape: tensor.shape().iter().map(|&d| d as _).collect(),
                data: &others.data[tensor.offset()..][..tensor.nbytes()],
            };
            if self.tensors.insert(name.to_string(), tensor).is_some() {
                return Err(GGufError::DuplicateTensorName(name.into()));
            }
        }
        Ok(())
    }
}

impl MetaValue<'_> {
    /// 从元信息读取 isize 数组，用于解析 strides
    pub fn to_vec_isize(&self) -> Result<Vec<isize>, GGufMetaError> {
        use GGufMetaDataValueType as Ty;

        let mut reader = GGufReader::new(&self.value);
        let (ty, len) = match self.ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };

        match ty {
            Ty::I32 => Ok(GGufMetaValueArray::<i32>::new(reader, len)
                .map(|x| x.unwrap() as _)
                .collect()),
            Ty::I64 => Ok(GGufMetaValueArray::<i64>::new(reader, len)
                .map(|x| x.unwrap() as _)
                .collect()),
            _ => Err(GGufMetaError::ArrTypeMismatch(ty)),
        }
    }
}
