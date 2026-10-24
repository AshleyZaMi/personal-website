import Product from "./Product.jsx"
// doing this because we will use the product for many product items inside of
// FeaturedProducts
import "./FeaturedProducts.css"

export default function FeaturedProducts(){
    return(
        <div className="featured-products-container" id="featured-products">
            <p className="section-header">Featured Products</p>
            <div className="products-container">
                {/* adding products */}
                <Product
                productImg = "./img/JapanTourTee.png"
                productAlt = "Japan Tour t-shirt"
                productName ="Japan Tour 2025 Tee"
                productPrice = {35}
                productRating = {10}
                ></Product>

                {/* Adding another product */}

                <Product
                productImg = "./img/SkullLogoHoodie.png"
                productAlt = "Skull Hoodie"
                productName="OOR Skull Logo Hoodie"
                productPrice = {75}
                productRating = {10}
                ></Product>

                <Product
                productImg = "./img/sapphireHoodie.png"
                productAlt = "sapphire hoodie"
                productName = "Sapphire Hoodie"
                productPrice = {85}
                productRating = {10}
                ></Product>

                <Product
                productImg = "./img/XAnniversarySweater.png"
                productAlt = "Tenth Aniversary Hoodie"
                productName ="X Aniversay Sweater"
                productPrice = {75}
                productRating = {10}
                ></Product>

            </div>
        </div>
    )
}