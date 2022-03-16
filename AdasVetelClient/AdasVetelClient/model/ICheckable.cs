namespace AdasVetelServer.model
{
    interface ICheckable<T>
    {
        bool equals(T element);
    }
}
