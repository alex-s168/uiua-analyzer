package me.alex_s168.uiua.mlir

object Inst {
    fun undef(dest: MLIRVar, type: MLIRType) =
        "$dest = llvm.mlir.undef : $type"

    fun insertValue(dest: MLIRVar, structType: MLIRType, value: MLIRVar, into: MLIRVar, idx: Int) =
        "$dest = llvm.insertvalue $value, $into [$idx] : $structType"

    fun extractValue(dest: MLIRVar, structType: MLIRType, value: MLIRVar, idx: Int) =
        "$dest = llvm.extractvalue $value [$idx] : $structType"

    fun call(dest: MLIRVar, type: MLIRType, fn: MLIRFn, vararg args: MLIRVar) =
        "$dest = func.call @${fn.name}(${args.joinToString()}) : (${fn.args.joinToString()}) -> $type"

    private fun baseBin(txt: String, dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        "$dest = " + (if (float) "arith.${txt}f $a, $b" else "arith.${txt}i $a, $b") + " : $type"

    fun add(dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        baseBin("add", dest, type, a, b, float)

    fun sub(dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        baseBin("sub", dest, type, a, b, float)

    fun mul(dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        baseBin("mul", dest, type, a, b, float)

    fun div(dest: MLIRVar, type: MLIRType, a: MLIRVar, b: MLIRVar, float: Boolean) =
        baseBin("div", dest, type, a, b, float)

    fun tensorExtract(dest: MLIRVar, tensorType: MLIRType, tensor: MLIRVar, vararg idx: MLIRVar) =
        "$dest = tensor.extract $tensor[${idx.joinToString()}] : $tensorType"

    fun tensorGenerate(
        dest: MLIRVar,
        tensorTy: MLIRType,
        dynamicDims: List<MLIRVar>,
        iterCords: List<MLIRVar>,
        iterRes: MLIRVar,
        iterResTy: MLIRType,
        inner: List<String>,
    ) = "$dest = tensor.generate ${dynamicDims.joinToString()} {\n^bb0(${iterCords.joinToString { "$it : index" }}):\n  ${inner.joinToString(separator = "\n  ")}\n  tensor.yield $iterRes : $iterResTy\n} : $tensorTy"

    fun tensorMemRef(dest: MLIRVar, memrefType: MLIRType, src: MLIRVar) =
        "$dest = bufferization.to_memref $src : $memrefType"
}
