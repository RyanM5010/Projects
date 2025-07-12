import { configureStore } from "@reduxjs/toolkit";
import { productReducer } from "./reducers/ProductReducer";
import { errortReducer } from "./reducers/ErrorReducer";
import { cartReducer } from "./reducers/cartReducer";
import { authReducer } from "./reducers/authReducer";
import { paymentMethodReducer } from "./reducers/paymentMethodReducer";

const cartItems = localStorage.getItem("cartItems")
    ? JSON.parse(localStorage.getItem("cartItems"))
    :[];

const user = localStorage.getItem("auth")
    ? JSON.parse(localStorage.getItem("auth"))
    : null;    

const selectedUserCheckoutAddress = localStorage.getItem("CHECKTOUT_ADDRESS")
    ? JSON.parse(localStorage.getItem("CHECKTOUT_ADDRESS"))
    : [];

const initialState = {
    auth: { user: user, selectedUserCheckoutAddress},
    carts:{ cart: cartItems },
};

export const store = configureStore({
    reducer:{
        products: productReducer,
        errors: errortReducer,
        carts: cartReducer,
        auth: authReducer,
        payment: paymentMethodReducer,
    },
    preloadedState: initialState,
});

export default store;