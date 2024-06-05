package me.alex_s168.uiua

import blitz.Either

// TODO: move into base blitz kt

fun unescape(str: String): String {
    val out = StringBuilder()
    var escaped = false

    for (char in str) {
        if (escaped) {
            escaped = false
            val e = when (char) {
                'n' -> '\n'
                'r' -> '\r'
                't' -> '\t'
                '\\' -> '\\'
                '"' -> '"'
                '\'' -> '\''
                else -> error("Unexpected character '$char'")
            }
            out.append(e)
        } else if (char == '\\') {
            escaped = true
        } else {
            out.append(char)
        }
    }

    return out.toString()
}

fun <T> MutableList<T>.addFront(value: T) =
    add(0, value)

fun <A, BA, BB, BAN> Either<A, Either<BA, BB>>.mapBA(fn: (BA) -> BAN): Either<A, Either<BAN, BB>> =
    mapB { it.mapA(fn) }

fun <A, BA, BB, BBN> Either<A, Either<BA, BB>>.mapBB(fn: (BB) -> BBN): Either<A, Either<BA, BBN>> =
    mapB { it.mapB(fn) }

fun <AA, AB, B, AAN> Either<Either<AA, AB>, B>.mapAA(fn: (AA) -> AAN): Either<Either<AAN, AB>, B> =
    mapA { it.mapA(fn) }

fun <AA, AB, B, ABN> Either<Either<AA, AB>, B>.mapAB(fn: (AB) -> ABN): Either<Either<AA, ABN>, B> =
    mapA { it.mapB(fn) }

fun <AA, AB, B> Either<Either<AA, AB>, B>.getAAOrNull(): AA? =
    getAOrNull()?.getAOrNull()

fun <AA, AB, B> Either<Either<AA, AB>, B>.getABOrNull(): AB? =
    getAOrNull()?.getBOrNull()

fun <A, BA, BB> Either<A, Either<BA, BB>>.getBAOrNull(): BA? =
    getBOrNull()?.getAOrNull()

fun <A, BA, BB> Either<A, Either<BA, BB>>.getBBOrNull(): BB? =
    getBOrNull()?.getBOrNull()