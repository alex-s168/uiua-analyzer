package me.alex_s168.uiua.mlir

object Inst {
    fun undef(dest: MLIRVar, type: MLIRType) =
        "$dest = llvm.mlir.undef : $type"

    fun insertValue(dest: MLIRVar, structType: MLIRType, value: MLIRVar, into: MLIRVar, idx: Int) =
        "$dest = llvm.insertvalue $value, $into [$idx] : $structType"

    fun extractValue(dest: MLIRVar, structType: MLIRType, value: MLIRVar, idx: Int) =
        "$dest = llvm.extractvalue $value [$idx] : $structType"

    fun call(dest: MLIRVar, type: MLIRType, fn: MLIRFn, vararg args: MLIRVar) =
        "$dest = func.call @${fn.name.legalizeMLIR()}(${args.joinToString()}) : (${fn.args.joinToString()}) -> $type"

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

    fun tensorExtract(
        dest: MLIRVar,
        tensorType: MLIRType,
        tensor: MLIRVar,
        vararg idx: MLIRVar
    ) = "$dest = tensor.extract $tensor[${idx.joinToString()}] : $tensorType"

    fun tensorGenerate(
        dest: MLIRVar,
        tensorTy: MLIRType,
        dynamicDims: List<MLIRVar>,
        iterCords: List<MLIRVar>,
        iterRes: MLIRVar,
        iterResTy: MLIRType,
        inner: List<String>,
    ) = "$dest = tensor.generate ${dynamicDims.joinToString()} {\n^bb0(${iterCords.joinToString { "$it : index" }}):\n  ${inner.joinToString(separator = "\n  ")}\n  tensor.yield $iterRes : $iterResTy\n} : $tensorTy"

    fun tensorMemRef(dest: MLIRVar, tensorType: MLIRType, src: MLIRVar) =
        "$dest = bufferization.to_memref $src : $tensorType"

    fun memRefTensor(dest: MLIRVar, memrefType: MLIRType, src: MLIRVar) =
        "$dest = bufferization.to_tensor $src : $memrefType"

    fun tensorDim(dest: MLIRVar, tensorType: MLIRType, tensor: MLIRVar, dim: MLIRVar) =
        "$dest = tensor.dim $tensor, $dim : $tensorType"

    fun memRefDim(dest: MLIRVar, memRefType: MLIRType, memRef: MLIRVar, dim: MLIRVar) =
        "$dest = memref.dim $memRef, $dim : $memRefType"

    fun memRefLoad(
        dest: MLIRVar,
        memRefType: MLIRType,
        memRef: MLIRVar,
        vararg idx: MLIRVar
    ) = "$dest = memref.load $memRef[${idx.joinToString()}] : $memRefType"

    fun memRefStore(
        memRefType: MLIRType,
        value: MLIRVar,
        memRef: MLIRVar,
        vararg idx: MLIRVar
    ) = "memref.store $value, $memRef[${idx.joinToString()}] : $memRefType"

    fun memRefAlloc(dest: MLIRVar, memRefType: MLIRType, vararg dim: MLIRVar) =
        "$dest = memref.alloc(${dim.joinToString()}) : $memRefType"

    fun affineParallelFor(counters: List<MLIRVar>, start: List<String>, end: List<String>, ops: List<String>) =
        "affine.parallel (${counters.joinToString()}) = (${start.joinToString()}) to (${end.joinToString()}) {\n  ${ops.joinToString("\n  ")}\n}"

    fun constant(dest: MLIRVar, type: MLIRType, value: String) =
        "$dest = arith.constant $value : $type"

    object Compound {
        fun memRefGenerate(
            dest: MLIRVar,
            memRefTy: MLIRType,
            allDims: List<MLIRVar>,
            dynDims: List<MLIRVar>,
            iterCords: List<MLIRVar>,
            iterRes: MLIRVar,
            iterResTy: MLIRType,
            inner: List<String>,
        ): List<String> {
            val res = mutableListOf<String>()
            res += memRefAlloc(dest, memRefTy, *dynDims.toTypedArray())
            val body = mutableListOf<String>()
            body.addAll(inner)
            body += memRefStore(memRefTy, iterRes, dest, *iterCords.toTypedArray())
            res += affineParallelFor(iterCords, allDims.map { "0" }, allDims, body)
            return res
        }
    }
}
