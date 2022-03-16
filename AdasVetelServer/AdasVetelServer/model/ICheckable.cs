namespace AdasVetelServer.model
{
    interface ICheckable<T>
    {
        bool valueEquals(T element);
    }
}
