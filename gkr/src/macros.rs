#[macro_export]
macro_rules! entered_span {
    ($first:expr $(,)*) => {
        $crate::tracing_span!($first).entered()
    };
}

#[macro_export]
macro_rules! tracing_span {
    ($first:expr $(,)*) => {
        tracing::span!(tracing::Level::DEBUG, $first)
    };
}

#[macro_export]
macro_rules! exit_span {
    ($first:expr $(,)*) => {
        $first.exit();
    };
}

#[macro_export]
#[cfg(feature = "parallel")]
macro_rules! izip_parallelizable {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::izip_parallizable!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        rayon::iter::IntoParallelIterator::into_par_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {
        $crate::izip_parallelizable!($first).zip($second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {
        $crate::izip_parallelizable!($first)
            $(.zip($rest))*
            .map($crate::izip_parallizable!(@closure a => (a) $(, $rest)*))
    };
}

#[macro_export]
#[cfg(not(feature = "parallel"))]
macro_rules! izip_parallelizable {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::izip_parallizable!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        $first.into_iter()
    };
    ($first:expr, $second:expr $(,)*) => {
        $crate::izip_parallelizable!($first).zip($second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {
        $crate::izip_parallelizable!($first)
            $(.zip($rest))*
            .map($crate::izip_parallizable!(@closure a => (a) $(, $rest)*))
    };
}
